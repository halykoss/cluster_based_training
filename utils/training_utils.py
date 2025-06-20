import numpy as np
import torch
import time
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, loader, optimizer, loss_fn, scheduler, max_steps=1000, val_loader=None, val_dataset=None, eval_interval=100):
    model.train()
    best_mse = float('inf')
    step = 0
    epoch_loss = 0.0
    samples_processed = 0
    start_time = time.time()
    
    # Create infinite data iterator
    def infinite_dataloader(dataloader):
        while True:
            for batch in dataloader:
                yield batch
    
    data_iter = infinite_dataloader(loader)
    
    while step < max_steps:
        X_batch, y_batch = next(data_iter)
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
        step += 1
        
        # Evaluation and logging
        if step % eval_interval == 0 or step == max_steps:
            epoch_time = time.time() - start_time
            
            if val_loader is not None and val_dataset is not None:
                model.eval()
                mse, _, _, _, _ = evaluate_model(model, val_loader, val_dataset)
                
                if mse < best_mse:
                    best_mse = mse
                
                model.train()
            
            # Reset counters for next interval
            epoch_loss = 0.0
            samples_processed = 0
            start_time = time.time()
    
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
