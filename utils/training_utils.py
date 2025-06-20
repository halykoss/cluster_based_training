import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error
from models import TransformerModelSinusoidal
from dataset.EncodedDataset import EncodedDataset
from tqdm import tqdm
import time
import os 

def train_model(trial, weights):
    device = torch.device("cuda:{}".format(trial.number % torch.cuda.device_count()) if torch.cuda.is_available() else "cpu")

    # Training hyperparameters
    batch_size = 1024
    learning_rate = 1e-4
    patch_size = 30

    # Model hyperparameters
    embed_dims = [256]
    num_heads_options = 8
    use_cnn_options = [True]

    # ==========================
    # 1. Prepare Dataset and DataLoader using EncodedDataset
    # ==========================

    # Use EncodedDataset which handles encoded data loading and cluster-based sampling
    # For now, we'll use the encoded data without cluster weights
    train_dataset = EncodedDataset(mode='train', use_encoded=False, include_clusters=True, cluster_weights=weights)
    if train_dataset.sampled_indices is not None:
        indices_dir = "data/indexes"
        if not os.path.exists(indices_dir):
            os.makedirs(indices_dir)
        np.save(os.path.join(indices_dir, "indices_trial_{}.npy".format(trial.number)), train_dataset.sampled_indices)
    test_dataset = EncodedDataset(mode='test', use_encoded=False, include_clusters=False)

    print(f"Training sequences count: {len(train_dataset)}")
    print(f"Test sequences count: {len(test_dataset)}")

    # Get data info for model configuration
    train_info = train_dataset.get_data_info()
    n_features = train_info['num_features']
    n_targets = train_info['num_targets']

    print(f"Number of features: {n_features}")
    print(f"Number of targets: {n_targets}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ==========================
    # 2. Define Models
    # ==========================
    # (Model definitions are imported from the models module)

    # ==========================
    # 3. Training and Evaluation Functions
    # ==========================
    def train_model(model, loader, optimizer, loss_fn, scheduler, epochs=10, val_loader=None, val_dataset=None):
        best_mse = float('inf')
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            samples_processed = 0
            start_time = time.time()  # Inizio misurazione del tempo
            
            for X_batch, y_batch in tqdm(loader, leave=False):
                batch_size = X_batch.size(0)
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                output = model(X_batch)
                loss = loss_fn(output, y_batch)
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item() * batch_size
                samples_processed += batch_size
            
            epoch_time = time.time() - start_time  # Calcolo tempo totale dell'epoca
            throughput = samples_processed / epoch_time  # Campioni processati al secondo
            epoch_loss /= len(loader.dataset)
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, LR: {current_lr:.6f}, Throughput: {throughput:.2f} samples/sec")
            
            if val_loader is not None and val_dataset is not None:
                model.eval()
                mse, rmse_dict, mae_dict, mape_dict, val_throughput = evaluate_model(model, val_loader, val_dataset)
                if mse < best_mse:
                    best_mse = mse
                print(f"Validation - Epoch {epoch+1} MSE: {mse:.4f}, Throughput: {val_throughput:.2f} samples/sec")
                for target_name, rmse in rmse_dict.items():
                    print(f"Validation - Epoch {epoch+1} RMSE for {target_name}: {rmse:.4f}")
                for target_name, mae in mae_dict.items():
                    print(f"Validation - Epoch {epoch+1} MAE for {target_name}: {mae:.4f}")
                for target_name, mape in mape_dict.items():
                    print(f"Validation - Epoch {epoch+1} MAPE for {target_name}: {mape:.2f}%")
                model.train()
        return best_mse

    def evaluate_model(model, loader, dataset):
        model.eval()
        preds, trues = [], []
        samples_processed = 0
        start_time = time.time()  # Inizio misurazione del tempo
        
        # Get scalers from the dataset
        scaler_x, scaler_y = dataset.get_scalers()
        target_names = dataset.get_target_names()
        
        with torch.no_grad():
            for X_batch, y_batch in loader:
                batch_size = X_batch.size(0)
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                output = model(X_batch)
                preds.append(output.cpu().numpy())
                trues.append(y_batch.cpu().numpy())
                samples_processed += batch_size
                
        eval_time = time.time() - start_time  # Calcolo tempo totale della valutazione
        throughput = samples_processed / eval_time if eval_time > 0 else 0  # Campioni processati al secondo
        
        preds = np.vstack(preds)
        trues = np.vstack(trues)
        preds_orig = scaler_y.inverse_transform(preds)
        trues_orig = scaler_y.inverse_transform(trues)
        mse = mean_squared_error(trues_orig, preds_orig)
        rmse_dict = {}
        mae_dict = {}
        mape_dict = {}
        for i, target_name in enumerate(target_names):
            rmse = np.sqrt(mean_squared_error(trues_orig[:, i], preds_orig[:, i]))
            mae = np.mean(np.abs(trues_orig[:, i] - preds_orig[:, i]))
            # Calcolo del MAPE con gestione dei valori zero o vicini a zero
            epsilon = 1e-10
            mape = np.mean(np.abs((trues_orig[:, i] - preds_orig[:, i]) / (np.abs(trues_orig[:, i]) + epsilon))) * 100
            
            rmse_dict[target_name] = rmse
            mae_dict[target_name] = mae
            mape_dict[target_name] = mape
        return mse, rmse_dict, mae_dict, mape_dict, throughput

    # ==========================
    # 4. Single Model Training (simplified from original model selection)
    # ==========================

    # Initialize a single model with fixed hyperparameters
    model_name = "TransformerSinusoidal"
    print(f"\nTraining {model_name} model...")

    # Create the model
    model = TransformerModelSinusoidal.TransformerModelSinusoidal(
        input_size=n_features, 
        output_size=n_targets, 
        patch_size=patch_size,
        embed_dim=embed_dims[0],
        num_heads=num_heads_options,
        use_cnn=use_cnn_options[0]
    )

    model = model.to(device)

    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = int(30 * train_dataset.dataset_ratio)
    print(f"Training for {epochs} epochs with dataset ratio {train_dataset.dataset_ratio:.2f}")
    total_steps = epochs * len(train_loader) 
    warmup_steps = max(1, int(0.3 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: float(step) / warmup_steps if step < warmup_steps else max(0.0, float(total_steps - step) / (total_steps - warmup_steps))
    )
    loss_fn = nn.MSELoss()

    # Train the model
    mse = train_model(model, train_loader, optimizer, loss_fn, scheduler, epochs=epochs, val_loader=test_loader, val_dataset=test_dataset)

    return mse
    
