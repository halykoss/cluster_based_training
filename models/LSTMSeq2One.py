import torch
import torch.nn as nn

class LSTMSeq2One(nn.Module):
    def __init__(self, input_dim=65, hidden_dim=100, output_dim=4):
        super().__init__()
        # Encoder: consumes (batch, 180, 65) → produces last hidden & cell states (1, batch, 100)
        self.encoder = nn.LSTM(input_size=input_dim,
                               hidden_size=hidden_dim,
                               batch_first=True)
        
        # Decoder: will consume one time‑step of size hidden_dim
        self.decoder = nn.LSTM(input_size=hidden_dim,
                               hidden_size=hidden_dim,
                               batch_first=True)
        
        # Final dense: (batch, 1, hidden_dim) → (batch, 1, output_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: Tensor of shape (batch, 180, 65)
        returns: Tensor of shape (batch, 1, 4)
        """
        # --- Encoder ---
        # enc_out is (batch, 180, hidden_dim) which we don’t use;
        # hn, cn are each (num_layers * num_directions, batch, hidden_dim)
        _, (hn, cn) = self.encoder(x)
        
        # squeeze away the layer dimension → (batch, hidden_dim)
        h = hn.squeeze(0)
        c = cn.squeeze(0)
        
        # --- RepeatVector (just one time‑step here) ---
        # make a sequence of length 1 from the final hidden state
        dec_input = h.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # --- Decoder ---
        # initialize decoder with encoder’s final states
        dec_out, _ = self.decoder(dec_input, (hn, cn))
        # dec_out is (batch, 1, hidden_dim)
        
        # --- Dense ---
        out = self.fc(dec_out)       # (batch, 1, output_dim)
        out = out.squeeze(1)         # (batch, output_dim)
        return out