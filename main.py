import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from models import TransformerModel, StackedLSTMNetwork, CNN7Model
from models import TransformerModelSinusoidal, TransformerModelRoPE
from models import LSTMSeq2One, BiLSTMSeq2One, AttentionLSTMEncoderDecoder
from dataset import TimeSeriesDataset
from tqdm import tqdm
import wandb
import time  # Aggiungo l'importazione del modulo time per misurare il throughput
import itertools  # Aggiungo itertools per generare combinazioni di iperparametri

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs=30
batch_size=1024
learning_rate=1e-4
patch_size=30
SEQ_LEN = 300  # Adjust as needed (e.g., 20 or 128)

# Definisco le griglie di ricerca per gli iperparametri
patch_sizes = [15, 30, 50]
embed_dims = [64, 128, 256]
num_heads_options = 8  # Fisso a 8 heads
use_cnn_options = [True, False]  # Aggiungo l'opzione use_cnn alla grid search

# Opzionalmente, limita le combinazioni da testare
# Se True, testiamo tutte le combinazioni possibili
# Se False, impostiamo valori predefiniti per i parametri non inclusi nel ciclo
test_all_combinations = True

# ==========================
# 1. Load Data
# ==========================
df = pd.read_csv("data/measures_v2.csv")

# ==========================
# 1a. Add the derived input features from the table
# ==========================
# Voltage magnitude
df['u_s'] = np.sqrt(df['u_d']**2 + df['u_q']**2)
# Current magnitude
df['i_s'] = np.sqrt(df['i_d']**2 + df['i_q']**2)
# Electric apparent power
df['S_el'] = 1.5 * df['u_s'] * df['i_s']
# Joint interactions
df['J1']  = df['i_s'] * df['motor_speed']
df['J2']  = df['S_el'] * df['motor_speed']

# ==========================
# 1b. Add EWM features at spans 200, 500, 1500, 4000
# ==========================
ewm_spans = [200, 500, 1500, 4000]
# we'll take the original measured + the newly derived columns
base_and_derived = [
    'u_q','coolant','u_d','motor_speed','i_d','i_q',
    'ambient','torque',
    'u_s','i_s','S_el','J1','J2'
]
for span in ewm_spans:
    for col in base_and_derived:
        df[f"{col}_ewm_{span}"] = df[col].ewm(span=span, adjust=False).mean()

# ==========================
# 2. Define which columns are your inputs and outputs
# ==========================
# start from the original nine:
measured = ['u_q','coolant','u_d','motor_speed','i_d','i_q','ambient','torque']
# add the five physics‐derived:
derived = ['u_s','i_s','S_el','J1','J2']
# add all the EWM cols
ewm_feats = [f"{col}_ewm_{span}" 
             for span in ewm_spans 
             for col in (measured + derived)]

features = measured + derived + ewm_feats

targets = ['stator_winding','stator_tooth','stator_yoke','pm']

# ==========================
# 2. Split Data by Profile ID
# ==========================
# Use profile_id==65 for test; the other profiles form the training set.
train_df = df[df['profile_id'] != 65].copy()
test_df = df[df['profile_id'] == 65].copy()

# ==========================
# 3. Normalize Data Using Training Set Statistics
# ==========================
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

train_df[features] = scaler_x.fit_transform(train_df[features])
train_df[targets] = scaler_y.fit_transform(train_df[targets])
test_df[features] = scaler_x.transform(test_df[features])
test_df[targets] = scaler_y.transform(test_df[targets])

# ==========================
# 4. Create Sequences Within Each Profile
# ==========================
# Set sequence length as desired. (The table used 128, but typically your data may allow a different value.)

# Le funzioni create_sequences_* non sono più necessarie poiché utilizzeremo 
# il nostro Dataset personalizzato per generare le sequenze on-demand

# ==========================
# 5. Prepare Dataset and DataLoader
# ==========================

# Utilizziamo il dataset modificato che genera sequenze on-demand
train_dataset = TimeSeriesDataset.ProfileSequenceDataset(train_df, features, targets, SEQ_LEN)
test_dataset = TimeSeriesDataset.ProfileSequenceDataset(test_df, features, targets, SEQ_LEN)

print(f"Training sequences count: {len(train_dataset)}")
print(f"Test sequences count: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==========================
# 6. Define Models
# ==========================

# ==========================
# 7. Training and Evaluation Functions
# ==========================
def train_model(model, loader, optimizer, loss_fn, scheduler, epochs=10, val_loader=None):
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
        
        if val_loader is not None:
            model.eval()
            mse, rmse_dict, mae_dict, mape_dict, val_throughput = evaluate_model(model, val_loader)
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

def evaluate_model(model, loader):
    model.eval()
    preds, trues = [], []
    samples_processed = 0
    start_time = time.time()  # Inizio misurazione del tempo
    
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
    for i, target_name in enumerate(targets):
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
# 8. Run Experiments with Each Model
# ==========================
results = {}
n_features = len(features)      # e.g., 9 input features (or adjust if using a different setting)
n_targets = len(targets)        # e.g., 3 targets

# Definiamo tutti i modelli che vogliamo addestrare
# Con relativi parametri specifici e flag per indicare se sono soggetti a tuning
all_models = {
    # Modelli non soggetti a tuning
    "LSTMSeq2One": {
        "class": LSTMSeq2One.LSTMSeq2One,
        "hyperparameter_tuning": False,
        "params": {"input_dim": n_features, "hidden_dim": 100, "output_dim": 4}
    },
    "BiLSTMSeq2One": {
        "class": BiLSTMSeq2One.BiLSTMSeq2One,
        "hyperparameter_tuning": False,
        "params": {"input_dim": n_features}
    },
    "AttentionLSTMEncoderDecoder": {
        "class": AttentionLSTMEncoderDecoder.AttentionLSTMEncoderDecoder,
        "hyperparameter_tuning": False,
        "params": {"input_dim": n_features}
    },
    "CNN": {
        "class": CNN7Model.CNN7Model,
        "hyperparameter_tuning": False,
        "params": {"input_dim": n_features, "seq_len": SEQ_LEN, "output_dim": n_targets}
    },
    "LSTM": {
        "class": StackedLSTMNetwork.StackedLSTMNetwork,
        "hyperparameter_tuning": False,
        "params": {"input_dim": n_features, "hidden_dim": 256, "output_dim": n_targets, "num_layers": 2}
    },
    # Modelli transformer soggetti a tuning degli iperparametri
    "Transformer": {
        "class": TransformerModel.TransformerModel,
        "hyperparameter_tuning": True,
        "params": {}
    },
    "TransformerSinusoidal": {
        "class": TransformerModelSinusoidal.TransformerModelSinusoidal,
        "hyperparameter_tuning": True,
        "params": {}
    },
    "TransformerRoPE": {
        "class": TransformerModelRoPE.TransformerModelRoPE,
        "hyperparameter_tuning": True,
        "params": {}
    }
}

# Genera tutte le combinazioni di iperparametri da testare
if test_all_combinations:
    hyperparams_combinations = list(itertools.product(patch_sizes, embed_dims, use_cnn_options))
else:
    # Combinazioni per testare singolarmente l'effetto di ogni parametro
    patch_combinations = [(p, embed_dims[1], False) for p in patch_sizes]
    embed_combinations = [(patch_sizes[1], e, False) for e in embed_dims]
    cnn_combinations = [(patch_sizes[1], embed_dims[1], c) for c in use_cnn_options]
    
    # Combina tutte le prove
    hyperparams_combinations = patch_combinations + embed_combinations + cnn_combinations

best_configs = {}  # Per memorizzare la migliore configurazione per ogni modello

# Un unico loop per tutti i modelli
for model_name, model_info in all_models.items():
    print(f"\nProcessing model: {model_name}")
    model_class = model_info["class"]
    
    # Se il modello richiede hyperparameter tuning
    if model_info["hyperparameter_tuning"]:
        best_mse = float('inf')
        best_config = None
        
        for patch_size, embed_dim, use_cnn in hyperparams_combinations:
            config_name = f"{model_name}_patch{patch_size}_embed{embed_dim}_cnn{use_cnn}"
            print(f"\nTraining {config_name}...")
            
            # Inizializza wandb con la configurazione degli iperparametri
            wandb.init(project="PSMS_experiment", name=config_name, config={
                "model": model_name,
                "patch_size": patch_size,
                "embed_dim": embed_dim,
                "num_heads": num_heads_options,
                "use_cnn": use_cnn,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "scheduler": "linear"
            }, reinit=True)
            
            # Crea il modello con gli iperparametri specificati
            model = model_class(
                input_size=n_features, 
                output_size=n_targets, 
                patch_size=patch_size,
                embed_dim=embed_dim,
                num_heads=num_heads_options,
                use_cnn=use_cnn
            )
            
            model = model.to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            total_steps = epochs * len(train_loader)
            warmup_steps = max(1, int(0.3 * total_steps))
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: float(step) / warmup_steps if step < warmup_steps else max(0.0, float(total_steps - step) / (total_steps - warmup_steps))
            )
            wandb.watch(model, log="all")
            loss_fn = nn.MSELoss()
            
            train_model(model, train_loader, optimizer, loss_fn, scheduler, epochs=epochs, val_loader=test_loader)
            mse, rmse_dict, mae_dict, mape_dict, test_throughput = evaluate_model(model, test_loader)
            
            # Registra i risultati finali
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
            
            # Aggiorna la migliore configurazione se necessario
            if mse < best_mse:
                best_mse = mse
                best_config = {
                    "patch_size": patch_size,
                    "embed_dim": embed_dim,
                    "use_cnn": use_cnn,
                    "mse": mse
                }
            
            wandb.finish()
        
        # Salva la migliore configurazione per questo modello
        best_configs[model_name] = best_config
        print(f"\nMigliore configurazione per {model_name}: {best_config}")
        
        # Aggiunge il modello con la migliore configurazione ai risultati
        results[model_name] = best_mse
    else:
        # Per i modelli che non necessitano di tuning degli iperparametri
        model_display_name = model_name
        print(f"\nTraining {model_display_name} model...")
        
        wandb.init(project="PSMS_experiment", name=model_display_name, config={
            "model": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "scheduler": "linear",
            **model_info["params"]  # Include i parametri specifici del modello
        }, reinit=True)
        
        # Crea il modello con i parametri predefiniti
        model = model_class(**model_info["params"])
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        total_steps = epochs * len(train_loader)
        warmup_steps = max(1, int(0.3 * total_steps))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: float(step) / warmup_steps if step < warmup_steps else max(0.0, float(total_steps - step) / (total_steps - warmup_steps))
        )
        wandb.watch(model, log="all")
        loss_fn = nn.MSELoss()
        
        train_model(model, train_loader, optimizer, loss_fn, scheduler, epochs=epochs, val_loader=test_loader)
        mse, rmse_dict, mae_dict, mape_dict, test_throughput = evaluate_model(model, test_loader)
        results[model_name] = mse
        
        # Log final throughput
        wandb.log({"final_test_throughput": test_throughput})
        print(f"Final test throughput: {test_throughput:.2f} samples/sec")
        
        wandb.finish()

print("\nMigliori configurazioni trovate:")
for model_name, config in best_configs.items():
    print(f"{model_name}: {config}")

print("\nFinal MSE Results:", results)