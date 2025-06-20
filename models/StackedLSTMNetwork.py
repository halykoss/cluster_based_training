import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# ----- LSTM Network -----
class StackedLSTMNetwork(nn.Module):
    """
    A stacked LSTM network with:
      - Two LSTM layers (hidden_dim=256, num_layers=2)
      - Three Dense layers: 
           FC1: 256 -> 128 with ReLU,
           FC2: 128 -> 32 with ReLU,
           FC3: 32 -> output_dim (linear)
      
    Only the LSTM output from the last time step is used for prediction.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(StackedLSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch, seq_len, hidden_dim)
        # Use the output from the last time step.
        last_out = lstm_out[:, -1, :]  # (batch, hidden_dim)
        x = F.relu(self.fc1(last_out))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x