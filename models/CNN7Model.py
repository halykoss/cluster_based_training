import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# ----- CNN Model as per your image -----
class CNN7Model(nn.Module):
    """
    CNN architecture based on your provided design:
      - Input expected as (batch, seq_len, features).
      - Two 1D Conv layers each with 64 filters and kernel_size=7.
      - A max pooling with kernel_size=2.
      - Flatten, then two FC layers (first with 32 units, then the output).
      
    The model accepts an extra parameter 'seq_len' so it can compute the flattened dimension.
    """
    def __init__(self, input_dim, seq_len, output_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=7)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Calculate the sequence length after the two conv layers and pooling.
        # With no padding and stride=1:
        # After conv1: length = seq_len - 7 + 1 = seq_len - 6
        # After conv2: length = (seq_len - 6) - 7 + 1 = seq_len - 12
        # After pooling: length = floor((seq_len - 12) / 2)
        conv_out_len = (seq_len - 12) // 2
        flattened_dim = 64 * conv_out_len
        
        self.fc1 = nn.Linear(flattened_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, features) => Permute to (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
