import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotaryPositionEmbedding(nn.Module):
    """
    Implementazione del Rotary Position Embedding (RoPE) come descritto nel paper
    "RoFormer: Enhanced Transformer with Rotary Position Embedding".
    """
    def __init__(self, embed_dim, max_seq_len=500):
        super(RotaryPositionEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Frequenze per la rotazione
        freq = 1.0 / (10000 ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        self.register_buffer('freq', freq)
        
        # Preparazione delle posizioni
        t = torch.arange(max_seq_len).type_as(freq)
        freqs = torch.outer(t, freq)  # [seq_len, dim/2]
        
        # Calcolo dei seni e coseni per la rotazione
        self.register_buffer('sin', freqs.sin())
        self.register_buffer('cos', freqs.cos())

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length ({seq_len}) exceeds maximum length ({self.max_seq_len})")
        
        # Prendi solo le dimensioni pari e dispari
        x_reshape = x.view(*x.shape[:-1], -1, 2)
        x1, x2 = x_reshape[..., 0], x_reshape[..., 1]
        
        # Seleziona i valori sin e cos appropriati per questa lunghezza di sequenza
        sin = self.sin[:seq_len].unsqueeze(0)  # [1, seq_len, embed_dim/2]
        cos = self.cos[:seq_len].unsqueeze(0)  # [1, seq_len, embed_dim/2]
        
        # Applica la rotazione
        x_out1 = x1 * cos - x2 * sin
        x_out2 = x2 * cos + x1 * sin
        
        # Ricomponi il tensore
        out = torch.stack([x_out1, x_out2], dim=-1)
        return out.flatten(-2)

class RotaryTransformerEncoderLayer(nn.Module):
    """
    Layer Transformer con RoPE integrato nell'attenzione
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(RotaryTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.rope = RotaryPositionEmbedding(d_model)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src):
        # Applica RoPE agli input - mantiene la forma (batch, seq_len, d_model)
        src_with_rope = self.rope(src)
        
        # MultiheadAttention si aspetta una forma (batch, seq_len, d_model)
        # Non è necessaria nessuna manipolazione aggiuntiva della dimensione
        attn_output, _ = self.self_attn(src_with_rope, src_with_rope, src_with_rope)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        
        # Feed-forward network con skip connection
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        
        return src

class TransformerModelRoPE(nn.Module):
    """
    Modello Transformer con Rotary Position Embedding
    
    Args:
        input_size (int): Numero di features nella serie temporale di input.
        output_size (int): Dimensione dell'output.
        patch_size (int, optional): Lunghezza di ogni patch. Default è 16.
        embed_dim (int, optional): Dimensione degli embedding delle patch. Default è 64.
        num_layers (int, optional): Numero di layer dell'encoder Transformer. Default è 2.
        num_heads (int, optional): Numero di attention heads in ogni layer Transformer. Default è 4.
        use_cnn (bool, optional): Se usare la CNN per la proiezione delle patch invece del layer lineare. Default è False.
        use_special_token (bool, optional): Se usare un token speciale per la regressione. Default è False.
    """
    def __init__(self, input_size, output_size, patch_size=16, embed_dim=64, num_layers=2, 
                num_heads=4, use_cnn=False, use_special_token=False):
        super(TransformerModelRoPE, self).__init__()
        self.patch_size = patch_size
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.use_cnn = use_cnn
        self.use_special_token = use_special_token
        
        # Modulo di proiezione: CNN o Linear
        if use_cnn:
            # Proiezione CNN: kernel_size = patch_size, stride = patch_size
            self.patch_projection = nn.Conv1d(
                in_channels=input_size,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size
            )
        else:
            # Proiezione lineare per le patch appiattite
            self.patch_projection = nn.Linear(patch_size * input_size, embed_dim)
        
        # Token speciale di regressione se richiesto
        if use_special_token:
            self.reg_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Aggiorna la lunghezza massima della sequenza per RoPE se necessario
        max_seq_len = 501 if use_special_token else 500
        
        # Layer encoder Transformer con RoPE
        self.layers = nn.ModuleList([
            RotaryTransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim*4)
            for _ in range(num_layers)
        ])
        
        # Layer fully-connected di output
        self.fc = nn.Linear(embed_dim, output_size)
    
    def forward(self, x):
        # x: (batch, sequence_length, input_size)
        B, L, D = x.shape
        
        if self.use_cnn:
            # Reshape per CNN (B, D, L)
            x = x.transpose(1, 2)
            # Applica proiezione CNN
            x_embedded = self.patch_projection(x)  # Output: (B, embed_dim, n_patches)
            # Reshape a (B, n_patches, embed_dim)
            x_embedded = x_embedded.transpose(1, 2)
        else:
            # Calcola il numero di patch complete che possono essere create
            n_patches = L // self.patch_size
            if n_patches == 0:
                raise ValueError(f"La lunghezza della sequenza ({L}) deve essere almeno pari a patch_size ({self.patch_size})")
            
            # Tronca x per avere un numero intero di patch
            x_truncated = x[:, :n_patches * self.patch_size, :]
            
            # Riorganizza in patch e appiattisci: (batch, n_patches, patch_size*input_size)
            x_patched = x_truncated.reshape(B, n_patches, self.patch_size * D)
            
            # Proietta le patch in embedding: (batch, n_patches, embed_dim)
            x_embedded = self.patch_projection(x_patched)
        
        # Aggiungi il token speciale di regressione se abilitato
        if self.use_special_token:
            reg_tokens = self.reg_token.expand(B, -1, -1)
            x_embedded = torch.cat([x_embedded, reg_tokens], dim=1)
        
        # Passa attraverso i layer Transformer con RoPE
        for layer in self.layers:
            x_embedded = layer(x_embedded)
        
        # Usa il token speciale per la previsione se abilitato, altrimenti usa l'ultimo token patch
        if self.use_special_token:
            x_out = self.fc(x_embedded[:, -1, :])  # Usa il token speciale (ultima posizione)
        else:
            x_out = self.fc(x_embedded[:, -1, :])  # Usa l'ultimo token patch
        
        return x_out
