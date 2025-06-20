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
import wandb
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
epochs = 30
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
train_dataset = EncodedDataset(mode='train', use_encoded=False, include_clusters=False)
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

        # Consolidate metrics for a single logging call
        metrics = {
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "lr": current_lr,
            "train_throughput": throughput  # Aggiungo il throughput alle metriche
        }

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, LR: {current_lr:.6f}, Throughput: {throughput:.2f} samples/sec")
        
        if val_loader is not None and val_dataset is not None:
            model.eval()
            mse, rmse_dict, mae_dict, mape_dict, val_throughput = evaluate_model(model, val_loader, val_dataset)
            print(f"Validation - Epoch {epoch+1} MSE: {mse:.4f}, Throughput: {val_throughput:.2f} samples/sec")
            metrics["val_mse"] = mse
            metrics["val_throughput"] = val_throughput  # Aggiungo il throughput in validazione
            for target_name, rmse in rmse_dict.items():
                print(f"Validation - Epoch {epoch+1} RMSE for {target_name}: {rmse:.4f}")
                metrics[f"val_rmse_{target_name}"] = rmse
            for target_name, mae in mae_dict.items():
                print(f"Validation - Epoch {epoch+1} MAE for {target_name}: {mae:.4f}")
                metrics[f"val_mae_{target_name}"] = mae
            for target_name, mape in mape_dict.items():
                print(f"Validation - Epoch {epoch+1} MAPE for {target_name}: {mape:.2f}%")
                metrics[f"val_mape_{target_name}"] = mape
            model.train()

        # Log all metrics at once
        wandb.log(metrics)

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

# Initialize wandb
wandb.init(project="PSMS_experiment", name=model_name, config={
    "model": model_name,
    "patch_size": patch_size,
    "embed_dim": embed_dims[0],
    "num_heads": num_heads_options,
    "use_cnn": use_cnn_options[0],
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "scheduler": "linear"
}, reinit=True)

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
total_steps = epochs * len(train_loader)
warmup_steps = max(1, int(0.3 * total_steps))
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: float(step) / warmup_steps if step < warmup_steps else max(0.0, float(total_steps - step) / (total_steps - warmup_steps))
)
wandb.watch(model, log="all")
loss_fn = nn.MSELoss()

# Train the model
train_model(model, train_loader, optimizer, loss_fn, scheduler, epochs=epochs, val_loader=test_loader, val_dataset=test_dataset)

# Final evaluation
mse, rmse_dict, mae_dict, mape_dict, test_throughput = evaluate_model(model, test_loader, test_dataset)

# Log final results
final_metrics = {
    "test_mse": mse,
    "test_throughput": test_throughput
}
for target, value in rmse_dict.items():
    final_metrics[f"test_rmse_{target}"] = value
for target, value in mae_dict.items():
    final_metrics[f"test_mae_{target}"] = value
for target, value in mape_dict.items():
    final_metrics[f"test_mape_{target}"] = value
    
wandb.log(final_metrics)

print(f"\nFinal Results:")
print(f"Test MSE: {mse:.4f}")
print(f"Test Throughput: {test_throughput:.2f} samples/sec")
for target_name, rmse in rmse_dict.items():
    print(f"RMSE for {target_name}: {rmse:.4f}")

wandb.finish()