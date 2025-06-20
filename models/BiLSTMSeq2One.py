import torch
import torch.nn as nn

class BiLSTMSeq2One(nn.Module):
    def __init__(self,
                 input_dim: int = 65,
                 enc_hidden: int = 100,
                 dec_hidden: int = 200,
                 output_dim: int = 4):
        super().__init__()
        # --- Encoder: bidirectional LSTM ---
        # input: (batch, 180, 65)
        # hidden_size per direction = enc_hidden
        # output features per time step = enc_hidden * 2 = 200
        self.encoder = nn.LSTM(input_size=input_dim,
                               hidden_size=enc_hidden,
                               batch_first=True,
                               bidirectional=True)
        
        # --- Decoder: single‐step LSTM ---
        # input_size = dec_hidden = 200
        # hidden_size       = dec_hidden = 200
        self.decoder = nn.LSTM(input_size=dec_hidden,
                               hidden_size=dec_hidden,
                               batch_first=True)
        
        # --- Dense layer to 4 outputs ---
        self.fc = nn.Linear(dec_hidden, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 180, 65)
        returns: (batch, 1, 4)
        """
        # --- 1) Encode ---
        # enc_out: (batch, 180, 2*enc_hidden) [we don’t use it here]
        # (h_n, c_n): each is (num_layers*2, batch, enc_hidden)
        _, (h_n, c_n) = self.encoder(x)
        
        # Split forward/backward for both hidden and cell:
        # h_n[0] = forward final hidden (batch, enc_hidden)
        # h_n[1] = backward final hidden (batch, enc_hidden)
        h_fwd, h_bwd = h_n[0], h_n[1]
        c_fwd, c_bwd = c_n[0], c_n[1]
        
        # --- 2) Concat-1 = hidden concat → (batch, 2*enc_hidden)= (β,200) ---
        h_cat = torch.cat([h_fwd, h_bwd], dim=1)
        
        # --- 3) Concat-2 = cell concat  → (batch, 2*enc_hidden)= (β,200) ---
        c_cat = torch.cat([c_fwd, c_bwd], dim=1)
        
        # --- 4) RepeatVector on the hidden concat ---
        # gives (batch, 1, 2*enc_hidden) = decoder’s Input‑1
        dec_input = h_cat.unsqueeze(1)
        
        # --- 5) Prepare decoder initial states ---
        #   h0 = h_cat.unsqueeze(0)  → (1, batch, dec_hidden)
        #   c0 = c_cat.unsqueeze(0)  → (1, batch, dec_hidden)
        h0 = h_cat.unsqueeze(0)
        c0 = c_cat.unsqueeze(0)
        
        # --- 6) Decode one step ---
        # dec_out: (batch, 1, dec_hidden)
        dec_out, _ = self.decoder(dec_input, (h0, c0))
        
        # --- 7) Final dense →
        # (batch, 1, dec_hidden) → (batch, 1, output_dim)
        out = self.fc(dec_out)
        out = out.squeeze(1)  # (batch, output_dim)
        return out