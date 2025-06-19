import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np


class ProfileSequenceDataset(Dataset):
    """Dataset che pre-genera tutte le sequenze in memoria per accesso più veloce."""
    
    def __init__(self, df, feature_cols, target_cols, seq_len):
        """
        Args:
            df: DataFrame contenente i dati di input
            feature_cols: Lista di colonne da utilizzare come features
            target_cols: Lista di colonne da utilizzare come target
            seq_len: Lunghezza delle sequenze
        """
        print(f"Pre-generating sequences in memory for faster training...")
        
        features = df[feature_cols].values
        targets = df[target_cols].values
        self.seq_len = seq_len
        
        # Tenere traccia dei profili per mantenere le sequenze all'interno dello stesso profilo
        profile_ids = df['profile_id'].values
        unique_profiles = np.unique(profile_ids)
        
        # Calcolare gli indici di inizio e fine per ogni profilo
        profile_boundaries = {}
        for profile_id in unique_profiles:
            profile_indices = np.where(profile_ids == profile_id)[0]
            profile_boundaries[profile_id] = (profile_indices[0], profile_indices[-1])
        
        # Pre-generare tutte le sequenze valide
        self.sequences = []
        self.targets_list = []
        
        for profile_id in unique_profiles:
            start_idx, end_idx = profile_boundaries[profile_id]
            # Una sequenza valida deve avere almeno seq_len+1 elementi
            if end_idx - start_idx >= seq_len:
                # Generare tutte le sequenze per questo profilo
                for i in range(start_idx, end_idx - seq_len + 1):
                    # X è una sequenza di seq_len elementi
                    X = features[i:i + seq_len]
                    # y è il valore target al time step successivo
                    y = targets[i + seq_len]
                    
                    self.sequences.append(X)
                    self.targets_list.append(y)
        
        # Convertire a tensori per accesso più veloce
        self.sequences = torch.tensor(np.array(self.sequences), dtype=torch.float32)
        self.targets_list = torch.tensor(np.array(self.targets_list), dtype=torch.float32)
        
        print(f"Generated {len(self.sequences)} sequences of length {seq_len}")
        print(f"Memory usage: {self.sequences.element_size() * self.sequences.nelement() / 1024**2:.2f} MB for features")
        print(f"Memory usage: {self.targets_list.element_size() * self.targets_list.nelement() / 1024**2:.2f} MB for targets")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Accesso diretto alle sequenze pre-generate."""
        return self.sequences[idx], self.targets_list[idx]