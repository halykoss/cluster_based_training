import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os


class EncodedDataset(Dataset):
    """
    Dataset per caricare i dati codificati da encoded_data/encoded_data.npz
    
    Supporta sia i dati originali che quelli codificati con MOMENT, 
    e include i cluster labels generati da KMeans.
    """
    
    def __init__(self, mode='train', use_encoded=False, include_clusters=True, 
                 cluster_weights=None, data_dir='encoded_data'):
        """
        Args:
            mode (str): 'train' o 'test' per specificare quale dataset usare
            use_encoded (bool): True per usare i dati codificati, False per i dati originali
            include_clusters (bool): True per includere i cluster labels nei dati restituiti
            cluster_weights (list): Lista di 36 pesi (0-1) per il weighted sampling dei cluster
            data_dir (str): directory contenente i file dei dati codificati
        """
        self.mode = mode
        self.use_encoded = use_encoded
        self.include_clusters = include_clusters
        self.cluster_weights = cluster_weights
        self.data_dir = data_dir
        
        # Verifica che i file esistano
        data_file = os.path.join(data_dir, 'encoded_data.npz')
        metadata_file = os.path.join(data_dir, 'scalers_and_metadata.pkl')
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"File dei dati non trovato: {data_file}")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"File metadata non trovato: {metadata_file}")
        
        # Carica i dati
        self._load_data()
        
        # Prepara il weighted sampling se richiesto
        if self.cluster_weights is not None:
            self._setup_weighted_sampling()
            sampled_indices = self.get_weighted_sample_indices()
            self.x_data = self.x_data[sampled_indices]
            self.y_data = self.y_data[sampled_indices]
            if self.cluster_labels is not None:
                self.cluster_labels = self.cluster_labels[sampled_indices]
    
    def _load_data(self):
        """Carica i dati dal file npz"""
        print(f"Caricamento dati {self.mode} ({'codificati' if self.use_encoded else 'originali'})...")
        
        # Carica i dati dal file npz
        data_npz = np.load(os.path.join(self.data_dir, 'encoded_data.npz'))
        
        # Carica i metadata
        with open(os.path.join(self.data_dir, 'scalers_and_metadata.pkl'), "rb") as f:
            self.metadata = pickle.load(f)
        
        # Seleziona i dati in base al mode
        if self.mode == 'train':
            if self.use_encoded:
                self.x_data = data_npz['encoded_x_train']
            else:
                self.x_data = data_npz['x_data_train']
            self.y_data = data_npz['y_data_train']
            # Carica i cluster labels se disponibili e richiesti
            if self.include_clusters and 'train_cluster_labels' in data_npz:
                self.cluster_labels = data_npz['train_cluster_labels']
            else:
                self.cluster_labels = None
        elif self.mode == 'test':
            if self.use_encoded:
                self.x_data = data_npz['encoded_x_test']
            else:
                self.x_data = data_npz['x_data_test']
            self.y_data = data_npz['y_data_test']
            # Carica i cluster labels se disponibili e richiesti
            if self.include_clusters and 'test_cluster_labels' in data_npz:
                self.cluster_labels = data_npz['test_cluster_labels']
            else:
                self.cluster_labels = None
        else:
            raise ValueError("mode deve essere 'train' o 'test'")
        
        print(f"Caricate {len(self.x_data)} sequenze")
        print(f"Dimensione input: {self.x_data.shape}")
        print(f"Dimensione target: {self.y_data.shape}")
        
        if self.cluster_labels is not None:
            print(f"Cluster labels: {len(self.cluster_labels)} etichette")
            print(f"Numero di cluster unici: {len(np.unique(self.cluster_labels))}")
            print(f"Distribuzione cluster: {np.bincount(self.cluster_labels)}")
        else:
            print("Nessun cluster label disponibile")
    
    def _setup_weighted_sampling(self):
        """Configura il weighted sampling basato sui cluster weights"""
        if self.cluster_labels is None:
            print("Warning: Cluster weights forniti ma nessun cluster label disponibile")
            self.weighted_indices = None
            return
        
        # Verifica che i pesi abbiano la lunghezza corretta
        n_clusters = len(np.unique(self.cluster_labels))
        if len(self.cluster_weights) != 36:
            raise ValueError(f"Devono essere forniti esattamente 36 pesi, ricevuti {len(self.cluster_weights)}")
        
        # Verifica che i pesi siano nel range [0, 1]
        if not all(0 <= w <= 1 for w in self.cluster_weights):
            raise ValueError("Tutti i pesi devono essere compresi tra 0 e 1")
        
        # Calcola quanti elementi prendere da ogni cluster
        self.cluster_indices = {}
        self.cluster_sample_counts = {}
        
        for cluster_id in range(36):
            # Trova tutti i campioni del cluster
            cluster_mask = self.cluster_labels == cluster_id
            cluster_sample_indices = np.where(cluster_mask)[0]
            
            if len(cluster_sample_indices) > 0:
                # Calcola quanti campioni prendere: numero_elementi * peso
                weight = self.cluster_weights[cluster_id]
                num_samples_to_take = int(len(cluster_sample_indices) * weight)
                
                self.cluster_indices[cluster_id] = cluster_sample_indices
                self.cluster_sample_counts[cluster_id] = num_samples_to_take
            else:
                self.cluster_indices[cluster_id] = np.array([])
                self.cluster_sample_counts[cluster_id] = 0
        
        print(f"Weighted sampling configurato:")
        print(f"- Pesi cluster: {self.cluster_weights}")
        print(f"- Distribuzione campioni per cluster:")
        total_original = 0
        total_weighted = 0
        for cluster_id in range(36):
            original_count = len(self.cluster_indices[cluster_id])
            weighted_count = self.cluster_sample_counts[cluster_id]
            total_original += original_count
            total_weighted += weighted_count
            if original_count > 0:
                weight = self.cluster_weights[cluster_id]
                print(f"  Cluster {cluster_id}: {original_count} campioni → {weighted_count} campioni (peso {weight})")
        print(f"- Totale campioni: {total_original} → {total_weighted}")
    
    def get_weighted_sample_indices(self):
        """
        Campiona elementi da ogni cluster rispettando i pesi configurati in _setup_weighted_sampling.
        Usa i cluster_sample_counts calcolati dai cluster_weights.
        
        Returns:
            np.array: indici campionati da tutti i cluster (shuffled)
        """
        if not hasattr(self, 'cluster_indices') or self.cluster_indices is None:
            # Fallback: restituisci tutti gli indici disponibili
            return np.arange(len(self.x_data))
        
        all_indices = []
        
        for cluster_id in range(36):
            cluster_samples = self.cluster_indices[cluster_id]
            target_samples = self.cluster_sample_counts[cluster_id]
            
            if len(cluster_samples) > 0 and target_samples > 0:
                # Campiona il numero di elementi specificato dai pesi
                samples_to_take = min(target_samples, len(cluster_samples))
                
                # Campiona randomicamente dal cluster
                if samples_to_take >= len(cluster_samples):
                    selected_indices = cluster_samples.copy()
                else:
                    selected_indices = np.random.choice(
                        cluster_samples, size=samples_to_take, replace=False
                    )
                
                all_indices.extend(selected_indices)
        
        # Mescola gli indici selezionati
        all_indices = np.array(all_indices)
        np.random.shuffle(all_indices)
        
        return all_indices
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        """
        Restituisce una tupla per l'indice specificato.
        Se include_clusters=True: (input, target, cluster_label)
        Se include_clusters=False: (input, target)
        """
        x = torch.tensor(self.x_data[idx], dtype=torch.float32)
        y = torch.tensor(self.y_data[idx], dtype=torch.float32)
        
        if self.include_clusters and self.cluster_labels is not None:
            cluster = torch.tensor(self.cluster_labels[idx], dtype=torch.long)
            return x, y, cluster
        else:
            return x, y
    
    def get_feature_names(self):
        """Restituisce i nomi delle feature"""
        return self.metadata.get('features', [])
    
    def get_target_names(self):
        """Restituisce i nomi dei target"""
        return self.metadata.get('targets', [])
    
    def get_scalers(self):
        """Restituisce gli scaler per feature e target"""
        return self.metadata.get('scaler_x'), self.metadata.get('scaler_y')
    
    def get_cluster_info(self):
        """Restituisce informazioni sui cluster"""
        if self.cluster_labels is not None:
            unique_clusters = np.unique(self.cluster_labels)
            cluster_counts = np.bincount(self.cluster_labels)
            return {
                'n_clusters': len(unique_clusters),
                'cluster_labels': unique_clusters,
                'cluster_counts': cluster_counts,
                'kmeans_model': self.metadata.get('kmeans_model'),
                'has_clusters': True
            }
        else:
            return {'has_clusters': False}
    
    def get_samples_by_cluster(self, cluster_id):
        """Restituisce gli indici delle sequenze appartenenti a un cluster specifico"""
        if self.cluster_labels is None:
            raise ValueError("Nessun cluster label disponibile")
        return np.where(self.cluster_labels == cluster_id)[0]
    
    def get_data_info(self):
        """Restituisce informazioni sui dati"""
        return {
            'num_sequences': len(self.x_data),
            'input_shape': self.x_data.shape,
            'target_shape': self.y_data.shape,
            'num_features': len(self.get_feature_names()) if not self.use_encoded else self.x_data.shape[-1],
            'num_targets': len(self.get_target_names()),
            'sequence_length': self.x_data.shape[1] if len(self.x_data.shape) > 2 else 1,
            'mode': self.mode,
            'use_encoded': self.use_encoded
        }
