import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error
from models import TransformerModelSinusoidal
from dataset.EncodedDataset import EncodedDataset
from utils.training_utils import train_model, evaluate_model
from tqdm import tqdm
import time
import optuna
import json
import os
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
max_steps = 30 * 1171  # Number of training steps instead of epochs
eval_interval = 1100  # Evaluate every 100 steps
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
def optimize(trial):
    cluster_weights = [trial.suggest_float(f"weight_{i}", 0.0, 1.0) for i in range(36)]
    
    print(f"Trial {trial.number}, campionamento dei pesi dei cluster:")
    print(f"Pesi: {cluster_weights}")
    
    # Prepare datasets with sampled cluster weights
    train_dataset = EncodedDataset(
        mode='train', 
        use_encoded=False, 
        include_clusters=True,
        cluster_weights=cluster_weights
    )
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
    total_steps = max_steps
    warmup_steps = max(1, int(0.3 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: float(step) / warmup_steps if step < warmup_steps else max(0.0, float(total_steps - step) / (total_steps - warmup_steps))
    )
    loss_fn = nn.MSELoss()

    # Train the model
    mse = train_model(model, train_loader, optimizer, loss_fn, scheduler, max_steps=max_steps, val_loader=test_loader, val_dataset=test_dataset, eval_interval=eval_interval)

    return mse

# ==========================
# 2. Funzione principale per l'ottimizzazione degli iperparametri
# ==========================
def run_cluster_weight_optimization(n_trials=50, study_name="cluster_weights_optimization", seed=42):
    """
    Esegue l'ottimizzazione dei pesi dei cluster utilizzando Optuna.
    
    Args:
        n_trials (int): Numero di trial di ottimizzazione da eseguire
        study_name (str): Nome dello studio Optuna
        seed (int): Seed per la riproducibilità
        
    Returns:
        dict: Dizionario con i pesi ottimali dei cluster e il relativo MSE
    """
    # Imposta i seed per la riproducibilità
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Crea la directory per salvare i risultati se non esiste
    results_dir = "optimization_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Creazione dello studio Optuna
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    )
    
    # Esecuzione dell'ottimizzazione
    print(f"Avvio ottimizzazione con {n_trials} trial...")
    study.optimize(optimize, n_trials=n_trials)
    
    # Estrai i migliori parametri e il miglior valore
    best_params = study.best_params
    best_value = study.best_value
    
    # Converti i parametri nel formato di pesi del cluster
    best_cluster_weights = [best_params[f"weight_{i}"] for i in range(36)]
    
    # Salva i risultati in un file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "best_mse": best_value,
        "best_cluster_weights": best_cluster_weights,
        "study_name": study_name,
        "n_trials": n_trials,
        "timestamp": timestamp
    }
    
    results_file = os.path.join(results_dir, f"optim_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(result, f, indent=4)
    
    print("\n" + "="*50)
    print(f"Ottimizzazione completata!")
    print(f"Miglior MSE: {best_value:.6f}")
    print(f"Migliori pesi dei cluster salvati in: {results_file}")
    print("="*50)
    
    return result

# ==========================
# 3. Visualizzazione dei risultati
# ==========================
def plot_cluster_weights(weights, mse, title=None):
    """
    Visualizza i pesi dei cluster come un grafico a barre.
    
    Args:
        weights (list): Lista dei pesi dei cluster
        mse (float): MSE associato ai pesi
        title (str): Titolo del grafico
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(14, 6))
    sns.set_style("whitegrid")
    
    # Crea il grafico a barre
    ax = sns.barplot(x=list(range(36)), y=weights)
    
    # Aggiungi etichette e titolo
    plt.xlabel("Cluster ID")
    plt.ylabel("Peso")
    plt.title(title or f"Pesi dei Cluster (MSE: {mse:.6f})")
    
    # Aggiungi i valori sopra le barre
    for i, v in enumerate(weights):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    
    # Salva il grafico
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"optimization_results/cluster_weights_{timestamp}.png", dpi=300)
    plt.show()

# ==========================
# 4. Esecuzione principale
# ==========================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ottimizzazione dei pesi dei cluster per il training")
    parser.add_argument("--trials", type=int, default=50, help="Numero di trial Optuna da eseguire")
    parser.add_argument("--study", type=str, default="cluster_weights_study", help="Nome dello studio Optuna")
    parser.add_argument("--seed", type=int, default=42, help="Seed per la riproducibilità")
    
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print(f"OTTIMIZZAZIONE DEI PESI DEI CLUSTER")
    print(f"Trials: {args.trials}")
    print(f"Study: {args.study}")
    print(f"Seed: {args.seed}")
    print("="*50 + "\n")
    
    # Esegui l'ottimizzazione
    result = run_cluster_weight_optimization(
        n_trials=args.trials,
        study_name=args.study,
        seed=args.seed
    )
    
    # Visualizza i risultati
    try:
        plot_cluster_weights(
            weights=result["best_cluster_weights"],
            mse=result["best_mse"],
            title=f"Pesi Ottimali dei Cluster (MSE: {result['best_mse']:.6f})"
        )
    except Exception as e:
        print(f"Errore nella visualizzazione dei pesi: {e}")
        print("I risultati sono comunque stati salvati nel file JSON.")