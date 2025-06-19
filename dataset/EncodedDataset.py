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
    
    def get_weighted_sample_indices(self, num_samples):
        """
        Restituisce indici campionati con weighted sampling per cluster
        
        Args:
            num_samples (int): numero di campioni da ottenere (ignorato, usa la somma dei campioni pesati)
            
        Returns:
            np.array: indici dei campioni selezionati
        """
        if not hasattr(self, 'cluster_indices') or self.cluster_indices is None:
            # Fallback a sampling uniforme
            return np.random.choice(len(self.x_data), size=num_samples, replace=True)
        
        selected_indices = []
        
        for cluster_id in range(36):
            cluster_samples = self.cluster_indices[cluster_id]
            num_to_sample = self.cluster_sample_counts[cluster_id]
            
            if num_to_sample > 0 and len(cluster_samples) > 0:
                # Campiona con replacement se necessario
                if num_to_sample <= len(cluster_samples):
                    # Campiona senza replacement
                    sampled_indices = np.random.choice(
                        cluster_samples, 
                        size=num_to_sample, 
                        replace=False
                    )
                else:
                    # Campiona con replacement se richiesti più campioni di quelli disponibili
                    sampled_indices = np.random.choice(
                        cluster_samples, 
                        size=num_to_sample, 
                        replace=True
                    )
                
                selected_indices.extend(sampled_indices)
        
        # Mescola gli indici selezionati
        selected_indices = np.array(selected_indices)
        np.random.shuffle(selected_indices)
        
        return selected_indices
    
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


class WeightedEncodedDataset(Dataset):
    """
    Dataset wrapper che implementa weighted sampling basato sui cluster.
    Genera campioni da ogni cluster basato sui pesi specificati.
    """
    
    def __init__(self, base_dataset):
        """
        Args:
            base_dataset (EncodedDataset): dataset base con weighted sampling configurato
        """
        self.base_dataset = base_dataset
        
        if not hasattr(base_dataset, 'cluster_indices') or base_dataset.cluster_indices is None:
            raise ValueError("Il dataset base deve avere il weighted sampling configurato")
        
        # Calcola la dimensione dell'epoch basata sui pesi
        self.epoch_size = sum(base_dataset.cluster_sample_counts.values())
        
        # Genera gli indici per l'epoch corrente
        self._generate_epoch_indices()
    
    def _generate_epoch_indices(self):
        """Genera nuovi indici per l'epoch corrente"""
        self.epoch_indices = self.base_dataset.get_weighted_sample_indices(self.epoch_size)
    
    def __len__(self):
        return len(self.epoch_indices)
    
    def __getitem__(self, idx):
        """Restituisce un campione usando l'indice mappato"""
        actual_idx = self.epoch_indices[idx]
        return self.base_dataset[actual_idx]
    
    def new_epoch(self):
        """Genera nuovi indici per una nuova epoch"""
        self._generate_epoch_indices()
    
    def get_current_cluster_distribution(self):
        """Restituisce la distribuzione dei cluster nell'epoch corrente"""
        if self.base_dataset.cluster_labels is None:
            return None
        
        current_clusters = self.base_dataset.cluster_labels[self.epoch_indices]
        return np.bincount(current_clusters, minlength=36)
    
    def get_epoch_info(self):
        """Restituisce informazioni sull'epoch corrente"""
        return {
            'epoch_size': len(self.epoch_indices),
            'expected_size': self.epoch_size,
            'cluster_distribution': self.get_current_cluster_distribution()
        }


def create_dataloader(mode='train', use_encoded=True, include_clusters=True, 
                     cluster_weights=None, batch_size=32, shuffle=None, 
                     num_workers=0, data_dir='encoded_data'):
    """
    Crea un DataLoader per i dati codificati
    
    Args:
        mode (str): 'train' o 'test'
        use_encoded (bool): True per usare dati codificati, False per dati originali
        include_clusters (bool): True per includere i cluster labels
        cluster_weights (list): Lista di 36 pesi (0-1) per weighted sampling per cluster
        batch_size (int): dimensione del batch
        shuffle (bool): se mescolare i dati (default: True per train, False per test)
        num_workers (int): numero di worker per il caricamento dati
        data_dir (str): directory contenente i dati
    
    Returns:
        DataLoader: dataloader configurato
    """
    if shuffle is None:
        shuffle = (mode == 'train')
    
    # Crea il dataset base
    base_dataset = EncodedDataset(
        mode=mode, 
        use_encoded=use_encoded, 
        include_clusters=include_clusters,
        cluster_weights=cluster_weights,
        data_dir=data_dir
    )
    
    # Se sono forniti i cluster weights, usa il WeightedEncodedDataset
    if cluster_weights is not None:
        dataset = WeightedEncodedDataset(base_dataset)
        # Per weighted sampling, disabilita shuffle del DataLoader
        # perché il sampling è già gestito dal WeightedEncodedDataset
        shuffle = False
    else:
        dataset = base_dataset
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader


def get_data_statistics(data_dir='encoded_data'):
    """
    Mostra statistiche sui dati codificati
    
    Args:
        data_dir (str): directory contenente i dati
    """
    try:
        # Carica dataset di train e test
        train_dataset_orig = EncodedDataset(mode='train', use_encoded=False, data_dir=data_dir)
        train_dataset_enc = EncodedDataset(mode='train', use_encoded=True, data_dir=data_dir)
        test_dataset_orig = EncodedDataset(mode='test', use_encoded=False, data_dir=data_dir)
        test_dataset_enc = EncodedDataset(mode='test', use_encoded=True, data_dir=data_dir)
        
        print("="*60)
        print("STATISTICHE DATASET CODIFICATO")
        print("="*60)
        
        print("\nDATI ORIGINALI:")
        print("-" * 30)
        train_info = train_dataset_orig.get_data_info()
        test_info = test_dataset_orig.get_data_info()
        
        print(f"Training set:")
        print(f"  - Sequenze: {train_info['num_sequences']}")
        print(f"  - Shape input: {train_info['input_shape']}")
        print(f"  - Shape target: {train_info['target_shape']}")
        print(f"  - Numero features: {train_info['num_features']}")
        print(f"  - Lunghezza sequenza: {train_info['sequence_length']}")
        
        print(f"\nTest set:")
        print(f"  - Sequenze: {test_info['num_sequences']}")
        print(f"  - Shape input: {test_info['input_shape']}")
        print(f"  - Shape target: {test_info['target_shape']}")
        
        print("\nDATI CODIFICATI (MOMENT):")
        print("-" * 30)
        train_info_enc = train_dataset_enc.get_data_info()
        test_info_enc = test_dataset_enc.get_data_info()
        
        print(f"Training set:")
        print(f"  - Sequenze: {train_info_enc['num_sequences']}")
        print(f"  - Shape embeddings: {train_info_enc['input_shape']}")
        print(f"  - Dimensione embedding: {train_info_enc['num_features']}")
        
        print(f"\nTest set:")
        print(f"  - Sequenze: {test_info_enc['num_sequences']}")
        print(f"  - Shape embeddings: {test_info_enc['input_shape']}")
        
        print("\nFEATURES E TARGET:")
        print("-" * 30)
        features = train_dataset_orig.get_feature_names()
        targets = train_dataset_orig.get_target_names()
        
        print(f"Features ({len(features)}):")
        for i, feature in enumerate(features[:10]):  # mostra solo le prime 10
            print(f"  {i+1:2d}. {feature}")
        if len(features) > 10:
            print(f"  ... e altre {len(features)-10} features")
        
        print(f"\nTargets ({len(targets)}):")
        for i, target in enumerate(targets):
            print(f"  {i+1}. {target}")
        
        # Informazioni sui cluster
        print("\nCLUSTER INFO:")
        print("-" * 30)
        train_cluster_info = train_dataset_orig.get_cluster_info()
        if train_cluster_info['has_clusters']:
            print(f"Numero di cluster: {train_cluster_info['n_clusters']}")
            print(f"Distribuzione cluster training: {train_cluster_info['cluster_counts'].tolist()}")
            
            test_cluster_info = test_dataset_orig.get_cluster_info()
            if test_cluster_info['has_clusters']:
                print(f"Distribuzione cluster test: {test_cluster_info['cluster_counts'].tolist()}")
        else:
            print("Nessun cluster disponibile")
        
        print("="*60)
        
    except Exception as e:
        print(f"Errore nel caricamento dei dati: {e}")


def test_dataloader(batch_size=4, data_dir='encoded_data'):
    """
    Testa il funzionamento del dataloader
    
    Args:
        batch_size (int): dimensione del batch per il test
        data_dir (str): directory contenente i dati
    """
    print("="*50)
    print("TEST DATALOADER")
    print("="*50)
    
    try:
        # Test con dati originali
        print("\n1. Test con dati originali:")
        train_loader_orig = create_dataloader(
            mode='train', 
            use_encoded=False, 
            batch_size=batch_size,
            data_dir=data_dir
        )
        
        # Prendi un batch
        for batch_idx, (x, y) in enumerate(train_loader_orig):
            print(f"   Batch {batch_idx + 1}:")
            print(f"   - Input shape: {x.shape}")
            print(f"   - Target shape: {y.shape}")
            print(f"   - Input range: [{x.min():.3f}, {x.max():.3f}]")
            print(f"   - Target range: [{y.min():.3f}, {y.max():.3f}]")
            break
        
        # Test con dati codificati
        print("\n2. Test con dati codificati (MOMENT):")
        train_loader_enc = create_dataloader(
            mode='train', 
            use_encoded=True, 
            include_clusters=True,
            batch_size=batch_size,
            data_dir=data_dir
        )
        
        # Prendi un batch
        for batch_idx, batch_data in enumerate(train_loader_enc):
            if len(batch_data) == 3:  # Con cluster labels
                x, y, clusters = batch_data
                print(f"   Batch {batch_idx + 1}:")
                print(f"   - Embeddings shape: {x.shape}")
                print(f"   - Target shape: {y.shape}")
                print(f"   - Cluster labels shape: {clusters.shape}")
                print(f"   - Embeddings range: [{x.min():.3f}, {x.max():.3f}]")
                print(f"   - Target range: [{y.min():.3f}, {y.max():.3f}]")
                print(f"   - Cluster range: [{clusters.min()}, {clusters.max()}]")
                print(f"   - Cluster labels in batch: {clusters.unique().tolist()}")
            else:  # Senza cluster labels
                x, y = batch_data
                print(f"   Batch {batch_idx + 1}:")
                print(f"   - Embeddings shape: {x.shape}")
                print(f"   - Target shape: {y.shape}")
                print(f"   - Embeddings range: [{x.min():.3f}, {x.max():.3f}]")
                print(f"   - Target range: [{y.min():.3f}, {y.max():.3f}]")
            break
        
        # Test set
        print("\n3. Test con test set:")
        test_loader = create_dataloader(
            mode='test', 
            use_encoded=True, 
            include_clusters=True,
            batch_size=batch_size,
            data_dir=data_dir
        )
        
        for batch_idx, batch_data in enumerate(test_loader):
            if len(batch_data) == 3:
                x, y, clusters = batch_data
                print(f"   Test batch:")
                print(f"   - Shape: {x.shape}")
                print(f"   - Cluster labels: {clusters.unique().tolist()}")
            else:
                x, y = batch_data
                print(f"   Test batch:")
                print(f"   - Shape: {x.shape}")
            print(f"   - Numero totale batch nel test set: {len(test_loader)}")
            break
        
        # Test specifico per cluster
        print("\n4. Test funzionalità cluster:")
        try:
            dataset = EncodedDataset(mode='train', use_encoded=True, include_clusters=True, data_dir=data_dir)
            cluster_info = dataset.get_cluster_info()
            
            if cluster_info['has_clusters']:
                print(f"   - Numero di cluster: {cluster_info['n_clusters']}")
                print(f"   - Cluster disponibili: {cluster_info['cluster_labels'].tolist()}")
                
                # Test per campioni di un cluster specifico
                first_cluster = cluster_info['cluster_labels'][0]
                samples_in_cluster = dataset.get_samples_by_cluster(first_cluster)
                print(f"   - Campioni nel cluster {first_cluster}: {len(samples_in_cluster)}")
            else:
                print("   - Nessun cluster disponibile")
        except Exception as e:
            print(f"   - Errore nel test cluster: {e}")
        
        print("\n✓ Tutti i test sono passati!")
        
    except Exception as e:
        print(f"\n✗ Errore nel test: {e}")
        import traceback
        traceback.print_exc()


def test_weighted_sampling(data_dir='encoded_data'):
    """
    Testa il weighted sampling basato sui cluster
    
    Args:
        data_dir (str): directory contenente i dati
    """
    print("="*50)
    print("TEST WEIGHTED SAMPLING")
    print("="*50)
    
    try:
        # Crea pesi esempio: favorisce alcuni cluster rispetto ad altri
        # Cluster 0-10: peso alto (0.8)
        # Cluster 11-25: peso medio (0.5)  
        # Cluster 26-35: peso basso (0.1)
        example_weights = [0.8] * 11 + [0.5] * 15 + [0.1] * 10
        
        print(f"Pesi esempio: {len(example_weights)} pesi")
        print(f"Primi 11 cluster (peso 0.8): {example_weights[:11]}")
        print(f"Cluster 11-25 (peso 0.5): {example_weights[11:26]}")
        print(f"Ultimi 10 cluster (peso 0.1): {example_weights[26:]}")
        
        # Test 1: Crea dataset con weighted sampling
        print("\n1. Test creazione dataset con weighted sampling:")
        train_loader = create_dataloader(
            mode='train',
            use_encoded=True,
            include_clusters=True,
            cluster_weights=example_weights,
            batch_size=32,
            data_dir=data_dir
        )
        
        print(f"   ✓ Dataloader creato con {len(train_loader)} batch")
        
        # Mostra informazioni sull'epoch
        if hasattr(train_loader.dataset, 'get_epoch_info'):
            epoch_info = train_loader.dataset.get_epoch_info()
            print(f"   - Dimensione epoch: {epoch_info['epoch_size']} campioni")
            print(f"   - Numero di batch: {len(train_loader)}")
        
        # Test 2: Analizza distribuzione cluster in un batch
        print("\n2. Test distribuzione cluster in batch:")
        for batch_idx, batch_data in enumerate(train_loader):
            if len(batch_data) == 3:
                x, y, clusters = batch_data
                unique_clusters, counts = torch.unique(clusters, return_counts=True)
                print(f"   Batch {batch_idx + 1}:")
                print(f"   - Cluster presenti: {unique_clusters.tolist()}")
                print(f"   - Conteggi: {counts.tolist()}")
                
                # Calcola distribuzione pesata teorica
                cluster_list = clusters.numpy()
                weighted_clusters = [c for c in cluster_list if example_weights[c] > 0.6]
                print(f"   - Cluster con peso alto (>0.6): {len(weighted_clusters)}/{len(cluster_list)} campioni")
            break
        
        # Test 3: Confronta distribuzione tra epoch diverse
        print("\n3. Test distribuzione tra epoch diverse:")
        if hasattr(train_loader.dataset, 'get_current_cluster_distribution'):
            dist1 = train_loader.dataset.get_current_cluster_distribution()
            
            # Genera nuova epoch
            train_loader.dataset.new_epoch()
            dist2 = train_loader.dataset.get_current_cluster_distribution()
            
            print(f"   Epoch 1 - cluster con più campioni: {np.argmax(dist1)}")
            print(f"   Epoch 2 - cluster con più campioni: {np.argmax(dist2)}")
            print(f"   Differenza distribuzione: {np.sum(np.abs(dist1 - dist2))}")
        
        # Test 4: Verifica che i pesi influenzino la distribuzione
        print("\n4. Test influenza pesi sulla distribuzione:")
        
        # Crea pesi estremi per vedere l'effetto
        extreme_weights = [0.0] * 36
        extreme_weights[0] = 1.0  # Solo cluster 0 ha peso
        
        try:
            extreme_loader = create_dataloader(
                mode='train',
                use_encoded=True,
                include_clusters=True,
                cluster_weights=extreme_weights,
                batch_size=32,
                data_dir=data_dir
            )
            
            # Verifica che prevalga il cluster 0
            cluster_counts = np.zeros(36)
            for batch_data in extreme_loader:
                if len(batch_data) == 3:
                    _, _, clusters = batch_data
                    for c in clusters:
                        cluster_counts[c] += 1
            
            most_common_cluster = np.argmax(cluster_counts)
            total_samples = cluster_counts.sum()
            print(f"   Con peso 1.0 solo su cluster 0:")
            print(f"   - Cluster più frequente: {most_common_cluster}")
            print(f"   - Totale campioni: {int(total_samples)}")
            if total_samples > 0:
                print(f"   - Percentuale cluster 0: {cluster_counts[0]/total_samples*100:.1f}%")
            else:
                print(f"   - Nessun campione generato")
            
        except Exception as e:
            print(f"   Warning: Test pesi estremi fallito: {e}")
        
        print("\n✓ Test weighted sampling completato!")
        
    except Exception as e:
        print(f"\n✗ Errore nel test weighted sampling: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Mostra statistiche sui dati
    get_data_statistics()
    
    # Testa il dataloader
    test_dataloader()
    
    # Testa il weighted sampling
    test_weighted_sampling()
