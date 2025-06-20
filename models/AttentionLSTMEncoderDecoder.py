import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLSTMEncoderDecoder(nn.Module):
    def __init__(self,
                 input_dim: int = 65,
                 hidden_dim: int = 100,
                 output_dim: int = 4):
        super().__init__()
        # Encoder: unidirectional LSTM
        #  input  → (batch, 180, 65)
        #  output → (batch, 180, hidden_dim)
        #  h_n, c_n → (1, batch, hidden_dim)
        self.encoder = nn.LSTM(input_size=input_dim,
                               hidden_size=hidden_dim,
                               batch_first=True)
        
        # Decoder: single‐step LSTM
        #  input_size = hidden_dim
        #  hidden_size = hidden_dim
        self.decoder = nn.LSTM(input_size=hidden_dim,
                               hidden_size=hidden_dim,
                               batch_first=True)
        
        # Final projection: (batch, 1, 2*hidden_dim) → (batch, 1, output_dim)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 180, 65)
        returns: (batch, 1, 4)
        """
        # 1) Encode
        # enc_out: (batch, 180, hidden_dim)
        # (h_n, c_n): each (1, batch, hidden_dim)
        enc_out, (h_n, c_n) = self.encoder(x)
        
        # squeeze away the layer‐dim → (batch, hidden_dim)
        h_final = h_n.squeeze(0)
        c_final = c_n.squeeze(0)
        
        # 2) RepeatVector: make decoder input of shape (batch, 1, hidden_dim)
        dec_input = h_final.unsqueeze(1)
        
        # 3) Decode one step, seeded with encoder’s final state
        #    dec_out: (batch, 1, hidden_dim)
        #    _ : (h_n_dec, c_n_dec) unused
        dec_out, _ = self.decoder(dec_input,
                                  (h_n, c_n))
        
        # 4) Dot‑1: score each encoder time‐step by dot(dec_out, enc_out)
        #    dec_out       : (batch, 1, hidden_dim)
        #    enc_out.swap  : (batch, hidden_dim, 180)
        #    scores        : (batch, 1, 180)
        scores = torch.bmm(dec_out, enc_out.transpose(1, 2))
        
        # 5) Softmax over the 180 positions → attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (batch, 1, 180)
        
        # 6) Dot‑2: context = weighted sum of encoder outputs
        #    context : (batch, 1, hidden_dim)
        context = torch.bmm(attn_weights, enc_out)
        
        # 7) Concat decoder output + context → (batch, 1, 2*hidden_dim)
        concat = torch.cat([dec_out, context], dim=2)
        
        # 8) Final Dense → (batch, 1, output_dim)
        out = self.fc(concat)
        out = out.squeeze(1) # (batch, output_dim)
        
        return out