import pandas as pd
import numpy as np
import torch
from momentfm import MOMENTPipeline
from tqdm import tqdm
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from dataset.TimeSeriesDataset import ProfileSequenceDataset

def encode_data_and_save():
    """
    Processo per codificare tutti i dati utilizzando MOMENT e salvarli in un file.
    """
    
    # ==========================
    # 1. Carica il modello MOMENT
    # ==========================
    print("Caricamento del modello MOMENT...")
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large", 
        model_kwargs={'task_name': 'embedding'},
    )
    model.init()
    print("Modello MOMENT caricato con successo!")
    
    # ==========================
    # 2. Carica e preprocessa i dati (come in main.py)
    # ==========================
    print("Caricamento dei dati...")
    df = pd.read_csv("data/measures_v2.csv")
    
    # Aggiungi le feature derivate
    df['u_s'] = np.sqrt(df['u_d']**2 + df['u_q']**2)
    df['i_s'] = np.sqrt(df['i_d']**2 + df['i_q']**2)
    df['S_el'] = 1.5 * df['u_s'] * df['i_s']
    df['J1'] = df['i_s'] * df['motor_speed']
    df['J2'] = df['S_el'] * df['motor_speed']
    
    # Aggiungi le feature EWM
    ewm_spans = [200, 500, 1500, 4000]
    base_and_derived = [
        'u_q','coolant','u_d','motor_speed','i_d','i_q',
        'ambient','torque',
        'u_s','i_s','S_el','J1','J2'
    ]
    for span in ewm_spans:
        for col in base_and_derived:
            df[f"{col}_ewm_{span}"] = df[col].ewm(span=span, adjust=False).mean()
    
    # Definisci le colonne delle feature
    measured = ['u_q','coolant','u_d','motor_speed','i_d','i_q','ambient','torque']
    derived = ['u_s','i_s','S_el','J1','J2']
    ewm_feats = [f"{col}_ewm_{span}" 
                 for span in ewm_spans 
                 for col in (measured + derived)]
    features = measured + derived + ewm_feats
    targets = ['stator_winding','stator_tooth','stator_yoke','pm']
    
    print(f"Totale features: {len(features)}")
    print(f"Totale targets: {len(targets)}")
    
    # ==========================
    # 3. Dividi i dati per profile_id
    # ==========================
    train_df = df[df['profile_id'] != 65].copy()
    test_df = df[df['profile_id'] == 65].copy()
    
    # ==========================
    # 4. Normalizza i dati
    # ==========================
    print("Normalizzazione dei dati...")
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    train_df[features] = scaler_x.fit_transform(train_df[features])
    train_df[targets] = scaler_y.fit_transform(train_df[targets])
    test_df[features] = scaler_x.transform(test_df[features])
    test_df[targets] = scaler_y.transform(test_df[targets])
    
    # ==========================
    # 5. Parametri per la codifica
    # ==========================
    SEQ_LEN = 300  # Lunghezza della sequenza come in main.py
    
    # ==========================
    # 6. Crea dataset utilizzando TimeSeriesDataset
    # ==========================
    print("Creazione dataset di training...")
    train_dataset = ProfileSequenceDataset(train_df, features, targets, SEQ_LEN)
    
    print("Creazione dataset di test...")
    test_dataset = ProfileSequenceDataset(test_df, features, targets, SEQ_LEN)
    
    print(f"Sequenze di training: {len(train_dataset)}")
    print(f"Sequenze di test: {len(test_dataset)}")
    
    # ==========================
    # 7. Estrai dati dai dataset
    # ==========================
    def extract_data_from_dataset(dataset):
        """Estrae tutti i dati da un dataset."""
        x_data = []
        y_data = []
        
        for i in range(len(dataset)):
            x, y = dataset[i]
            x_data.append(x.numpy())
            y_data.append(y.numpy())
        
        return np.array(x_data), np.array(y_data)
    
    print("Estrazione dati di training...")
    x_data_train, y_data_train = extract_data_from_dataset(train_dataset)
    
    print("Estrazione dati di test...")
    x_data_test, y_data_test = extract_data_from_dataset(test_dataset)
    
    # ==========================
    # 8. Codifica con MOMENT
    # ==========================
    def encode_sequences(sequences, model, batch_size=32):
        """Codifica le sequenze utilizzando MOMENT."""
        encoded_sequences = []
        
        # Processa in batch per gestire grandi quantità di dati
        for i in tqdm(range(0, len(sequences), batch_size), desc="Codifica sequenze"):
            batch = sequences[i:i + batch_size]
            
            # Converti in tensor PyTorch se non lo è già
            if not isinstance(batch, torch.Tensor):
                batch_tensor = torch.tensor(batch, dtype=torch.float32)
            else:
                batch_tensor = batch
            
            # Codifica con MOMENT
            with torch.no_grad():
                embeddings = model(x_enc=batch_tensor)
                if hasattr(embeddings, 'last_hidden_state'):
                    embeddings = embeddings.last_hidden_state
                elif hasattr(embeddings, 'embeddings'):
                    embeddings = embeddings.embeddings
            
            encoded_sequences.append(embeddings.cpu().numpy())
        
        return np.vstack(encoded_sequences)
    
    print("Codifica dati di training...")
    encoded_x_train = encode_sequences(x_data_train, model)
    
    print("Codifica dati di test...")
    encoded_x_test = encode_sequences(x_data_test, model)
    
    # ==========================
    # 9. Clustering KMeans sulle embeddings
    # ==========================
    print("Esecuzione KMeans clustering sulle embeddings...")
    n_clusters = 36
    
    # Reshape delle embeddings per il clustering (flatten ogni sequenza)
    # Le embeddings hanno shape (n_sequences, seq_len, embedding_dim)
    # Per KMeans le convertiamo in (n_sequences, seq_len * embedding_dim)
    encoded_x_train_flat = encoded_x_train.reshape(encoded_x_train.shape[0], -1)
    encoded_x_test_flat = encoded_x_test.reshape(encoded_x_test.shape[0], -1)
    
    # Fit KMeans sui dati di training
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train_cluster_labels = kmeans.fit_predict(encoded_x_train_flat)
    
    # Predici i cluster per i dati di test
    test_cluster_labels = kmeans.predict(encoded_x_test_flat)
    
    print(f"Clustering completato con {n_clusters} cluster")
    print(f"Distribuzione cluster training: {np.bincount(train_cluster_labels)}")
    print(f"Distribuzione cluster test: {np.bincount(test_cluster_labels)}")
    
    # ==========================
    # 10. Salva i dati codificati e i cluster
    # ==========================
    # ==========================
    # 10. Salva i dati codificati e i cluster
    # ==========================
    print("Salvataggio dei dati codificati e cluster...")
    
    # Crea la directory di output se non esiste
    os.makedirs("encoded_data", exist_ok=True)
    
    # Salva i dati con i nomi specificati
    np.savez_compressed("encoded_data/encoded_data.npz",
                       x_data_train=x_data_train,
                       y_data_train=y_data_train,
                       encoded_x_train=encoded_x_train,
                       train_cluster_labels=train_cluster_labels,
                       x_data_test=x_data_test,
                       y_data_test=y_data_test,
                       encoded_x_test=encoded_x_test,
                       test_cluster_labels=test_cluster_labels)
    
    # Salva anche gli scaler, KMeans e metadata
    scalers_and_models = {
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'features': features,
        'targets': targets,
        'kmeans_model': kmeans,
        'n_clusters': n_clusters
    }
    
    with open("encoded_data/scalers_and_metadata.pkl", "wb") as f:
        pickle.dump(scalers_and_models, f)
    
    # Salva anche i metadata delle feature come file di testo per facilità di lettura
    with open("encoded_data/features_info.txt", "w") as f:
        f.write("FEATURES LIST:\n")
        f.write("="*50 + "\n")
        for i, feature in enumerate(features):
            f.write(f"{i:3d}: {feature}\n")
        f.write("\nTARGETS LIST:\n")
        f.write("="*50 + "\n")
        for i, target in enumerate(targets):
            f.write(f"{i:3d}: {target}\n")
        f.write(f"\nCLUSTERING INFO:\n")
        f.write("="*50 + "\n")
        f.write(f"Numero di cluster: {n_clusters}\n")
        f.write(f"Algoritmo: KMeans\n")
        f.write(f"Distribuzione cluster training: {np.bincount(train_cluster_labels).tolist()}\n")
        f.write(f"Distribuzione cluster test: {np.bincount(test_cluster_labels).tolist()}\n")
    
    # ==========================
    # 11. Stampa statistiche finali
    # ==========================
    print("\n" + "="*50)
    print("CODIFICA E CLUSTERING COMPLETATI!")
    print("="*50)
    print(f"Sequenze di training codificate: {len(x_data_train)}")
    print(f"Sequenze di test codificate: {len(x_data_test)}")
    print(f"Dimensione input originale: {x_data_train.shape}")
    print(f"Dimensione target: {y_data_train.shape}")
    print(f"Dimensione embeddings train: {encoded_x_train.shape}")
    print(f"Dimensione embeddings test: {encoded_x_test.shape}")
    print(f"Numero di cluster: {n_clusters}")
    print(f"Distribuzione cluster training: {np.bincount(train_cluster_labels)}")
    print(f"Distribuzione cluster test: {np.bincount(test_cluster_labels)}")
    print(f"Lunghezza sequenza: {SEQ_LEN}")
    print(f"Numero feature originali: {len(features)}")
    print(f"Numero target: {len(targets)}")
    print("\nFile salvati in:")
    print("- encoded_data/encoded_data.npz")
    print("  - x_data_train: sequenze input di training")
    print("  - y_data_train: target di training")
    print("  - encoded_x_train: embeddings MOMENT di training")
    print("  - train_cluster_labels: etichette cluster di training")
    print("  - x_data_test: sequenze input di test")
    print("  - y_data_test: target di test")
    print("  - encoded_x_test: embeddings MOMENT di test")
    print("  - test_cluster_labels: etichette cluster di test")
    print("- encoded_data/scalers_and_metadata.pkl")
    print("- encoded_data/features_info.txt")
    print("="*50)
    
    return {
        'x_data_train': x_data_train,
        'y_data_train': y_data_train,
        'encoded_x_train': encoded_x_train,
        'train_cluster_labels': train_cluster_labels,
        'x_data_test': x_data_test,
        'y_data_test': y_data_test,
        'encoded_x_test': encoded_x_test,
        'test_cluster_labels': test_cluster_labels
    }, scalers_and_models

def load_encoded_data():
    """
    Carica i dati codificati e i cluster precedentemente salvati da file npz.
    """
    print("Caricamento dati codificati e cluster...")
    
    # Carica i dati dal file npz
    data_npz = np.load("encoded_data/encoded_data.npz")
    data = {
        'x_data_train': data_npz['x_data_train'],
        'y_data_train': data_npz['y_data_train'],
        'encoded_x_train': data_npz['encoded_x_train'],
        'train_cluster_labels': data_npz['train_cluster_labels'],
        'x_data_test': data_npz['x_data_test'],
        'y_data_test': data_npz['y_data_test'],
        'encoded_x_test': data_npz['encoded_x_test'],
        'test_cluster_labels': data_npz['test_cluster_labels']
    }
    
    # Carica i metadata (ancora in pickle per compatibilità con gli oggetti scaler e KMeans)
    with open("encoded_data/scalers_and_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    print(f"Caricati {len(data['x_data_train'])} sequenze di training")
    print(f"Caricati {len(data['x_data_test'])} sequenze di test")
    print(f"Dimensione input originale: {data['x_data_train'].shape}")
    print(f"Dimensione embeddings train: {data['encoded_x_train'].shape}")
    print(f"Dimensione embeddings test: {data['encoded_x_test'].shape}")
    print(f"Numero di cluster: {metadata.get('n_clusters', 'N/A')}")
    if 'train_cluster_labels' in data:
        print(f"Distribuzione cluster training: {np.bincount(data['train_cluster_labels'])}")
        print(f"Distribuzione cluster test: {np.bincount(data['test_cluster_labels'])}")
    
    return data, metadata

if __name__ == "__main__":
    # Esegui la codifica completa
    encoded_data, metadata = encode_data_and_save()

