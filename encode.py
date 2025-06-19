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

def print_gpu_memory_usage():
    """Stampa l'utilizzo attuale della memoria GPU"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        print(f"Memoria GPU: {allocated:.2f} GB allocata, {cached:.2f} GB riservata")
    else:
        print("CUDA non disponibile")

def encode_data_and_save():
    """
    Processo per codificare tutti i dati utilizzando MOMENT e salvarli in un file.
    """
    
    # ==========================
    # 0. Configurazione Multi-GPU
    # ==========================
    if not torch.cuda.is_available():
        print("CUDA non disponibile, utilizzo CPU")
        device = torch.device("cpu")
        use_multi_gpu = False
    else:
        num_gpus = torch.cuda.device_count()
        print(f"GPU disponibili: {num_gpus}")
        
        # Mostra informazioni per tutte le GPU
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memoria: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
        
        if num_gpus >= 4:
            print("Utilizzo 4 GPU per il data parallelism")
            use_multi_gpu = True
            device = torch.device("cuda:0")  # GPU principale
        elif num_gpus > 1:
            print(f"Utilizzo {num_gpus} GPU per il data parallelism")
            use_multi_gpu = True
            device = torch.device("cuda:0")  # GPU principale
        else:
            print("Una sola GPU disponibile, utilizzo singola GPU")
            use_multi_gpu = False
            device = torch.device("cuda:0")
    
    # ==========================
    # 1. Carica il modello MOMENT
    # ==========================
    print("Caricamento del modello MOMENT...")
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large", 
        model_kwargs={'task_name': 'embedding'},
    )
    model.init()
    
    # Sposta il modello su GPU e configura DataParallel se necessario
    model = model.to(device)
    
    if use_multi_gpu:
        # Determina quali GPU utilizzare
        if torch.cuda.device_count() >= 4:
            gpu_ids = [0, 1, 2, 3]  # Usa le prime 4 GPU
        else:
            gpu_ids = list(range(torch.cuda.device_count()))  # Usa tutte le GPU disponibili
        
        print(f"Configurazione DataParallel su GPU: {gpu_ids}")
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        print(f"Modello MOMENT caricato con DataParallel su {len(gpu_ids)} GPU!")
    else:
        print(f"Modello MOMENT caricato su {device}!")
    
    # Monitora memoria GPU dopo caricamento modello
    if device.type == 'cuda':
        print("Memoria GPU dopo caricamento modello:")
        for i in range(min(4, torch.cuda.device_count())):
            print(f"  GPU {i}: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB allocata")
    
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
    def encode_sequences(sequences, model, device, use_multi_gpu, batch_size=None):
        """Codifica le sequenze utilizzando MOMENT con supporto Multi-GPU."""
        
        # Calcola batch size ottimale per multi-GPU
        if batch_size is None:
            if use_multi_gpu:
                num_gpus = len(model.device_ids) if hasattr(model, 'device_ids') else torch.cuda.device_count()
                # Batch size più grande per multiple GPU
                base_batch_size = 64
                batch_size = base_batch_size * num_gpus
                print(f"Multi-GPU: utilizzo {num_gpus} GPU con batch_size totale {batch_size}")
            elif device.type == 'cuda':
                # Singola GPU
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                if gpu_memory_gb >= 24:
                    batch_size = 128
                elif gpu_memory_gb >= 12:
                    batch_size = 64
                else:
                    batch_size = 32
            else:
                # CPU
                batch_size = 16
        
        print(f"Utilizzando batch_size: {batch_size}")
        encoded_sequences = []
        
        # Processa in batch per gestire grandi quantità di dati
        for i in tqdm(range(0, len(sequences), batch_size), desc="Codifica sequenze"):
            batch = sequences[i:i + batch_size]
            
            # Converti in tensor PyTorch se non lo è già
            if not isinstance(batch, torch.Tensor):
                batch_tensor = torch.tensor(batch, dtype=torch.float32)
            else:
                batch_tensor = batch
            
            # Sposta il batch su GPU/CPU
            batch_tensor = batch_tensor.to(device)
            
            # Codifica con MOMENT
            with torch.no_grad():
                embeddings = model(x_enc=batch_tensor)
                if hasattr(embeddings, 'last_hidden_state'):
                    embeddings = embeddings.last_hidden_state
                elif hasattr(embeddings, 'embeddings'):
                    embeddings = embeddings.embeddings
            
            # Sposta i risultati su CPU per salvare memoria GPU
            encoded_sequences.append(embeddings.cpu().numpy())
            
            # Pulisci la cache GPU se necessario per tutte le GPU
            if device.type == 'cuda':
                if use_multi_gpu:
                    # Pulisci cache per tutte le GPU utilizzate
                    for gpu_id in range(min(4, torch.cuda.device_count())):
                        torch.cuda.empty_cache(device=gpu_id)
                else:
                    torch.cuda.empty_cache()
        
        return np.vstack(encoded_sequences)
    
    print("Codifica dati di training...")
    if device.type == 'cuda':
        if use_multi_gpu:
            print("Memoria GPU prima della codifica training:")
            for i in range(min(4, torch.cuda.device_count())):
                print(f"  GPU {i}: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB allocata")
        else:
            print_gpu_memory_usage()
    encoded_x_train = encode_sequences(x_data_train, model, device, use_multi_gpu)
    
    print("Codifica dati di test...")
    if device.type == 'cuda':
        if use_multi_gpu:
            print("Memoria GPU prima della codifica test:")
            for i in range(min(4, torch.cuda.device_count())):
                print(f"  GPU {i}: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB allocata")
        else:
            print_gpu_memory_usage()
    encoded_x_test = encode_sequences(x_data_test, model, device, use_multi_gpu)
    
    # Libera il modello dalla memoria GPU dopo la codifica
    if device.type == 'cuda':
        del model
        if use_multi_gpu:
            # Pulisci cache per tutte le GPU utilizzate
            for gpu_id in range(min(4, torch.cuda.device_count())):
                torch.cuda.empty_cache(device=gpu_id)
            print("Memoria GPU liberata dopo la codifica (multi-GPU)")
        else:
            torch.cuda.empty_cache()
            print("Memoria GPU liberata dopo la codifica")
    
    # ==========================
    # 9. Clustering KMeans sulle embeddings
    # ==========================
    print("Esecuzione KMeans clustering sulle embeddings...")
    n_clusters = 36
    
    # Reshape delle embeddings per il clustering (flatten ogni sequenza)
    # Le embeddings hanno shape (n_sequences, seq_len, embedding_dim)
    # Per KMeans le convertiamo in (n_sequences, seq_len * embedding_dim)
    print("Preparazione dati per clustering...")
    encoded_x_train_flat = encoded_x_train.reshape(encoded_x_train.shape[0], -1)
    encoded_x_test_flat = encoded_x_test.reshape(encoded_x_test.shape[0], -1)
    
    print(f"Dimensione dati flattened per training: {encoded_x_train_flat.shape}")
    print(f"Dimensione dati flattened per test: {encoded_x_test_flat.shape}")
    
    # Fit KMeans sui dati di training
    print("Avvio KMeans clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train_cluster_labels = kmeans.fit_predict(encoded_x_train_flat)
    
    # Predici i cluster per i dati di test
    print("Predizione cluster per dati di test...")
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
    print(f"Dispositivo utilizzato: {device}")
    if device.type == 'cuda':
        if use_multi_gpu:
            num_gpus_used = min(4, torch.cuda.device_count())
            print(f"Multi-GPU utilizzate: {num_gpus_used} GPU")
            for i in range(num_gpus_used):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print("Memoria GPU finale:")
            for i in range(num_gpus_used):
                print(f"  GPU {i}: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB allocata")
        else:
            print(f"GPU utilizzata: {torch.cuda.get_device_name(0)}")
            print_gpu_memory_usage()
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

