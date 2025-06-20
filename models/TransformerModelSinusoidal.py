import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Implementazione della codifica posizionale sinusoidale come descritto
    nel paper "Attention Is All You Need".
    """
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]

class TransformerModelSinusoidal(nn.Module):
    """
    PatchTST model con codifica posizionale sinusoidale:
      - Divide la serie temporale di input in patch di lunghezza fissa.
      - Ogni patch viene appiattita e proiettata in uno spazio di embedding.
      - Viene aggiunta una codifica posizionale sinusoidale agli embedding delle patch.
      - Un encoder Transformer elabora i token delle patch.
      - La rappresentazione dell'ultima patch viene utilizzata per la previsione.
    
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
        super(TransformerModelSinusoidal, self).__init__()
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
            # Proiezione lineare per le patch appiattite (patch_size * input_size -> embed_dim)
            self.patch_projection = nn.Linear(patch_size * input_size, embed_dim)
        
        # Token speciale di regressione se richiesto
        if use_special_token:
            self.reg_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            
        # Codifica posizionale sinusoidale - aumentata di 1 per il token speciale se usato
        max_len = 501 if use_special_token else 500
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=max_len)
        
        # Encoder Transformer con batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer fully-connected di output
        self.fc = nn.Linear(embed_dim, output_size)
    
    def forward(self, x):
        # x: (batch, sequence_length, input_size)
        B, L, D = x.shape
        
        if self.use_cnn:
            # Reshape per CNN (B, D, L)
            x = x.transpose(1, 2)
            # Applica proiezione CNN
            x = self.patch_projection(x)  # Output: (B, embed_dim, n_patches)
            # Reshape a (B, n_patches, embed_dim)
            x = x.transpose(1, 2)
            n_patches = x.shape[1]
        else:
            n_patches = L // self.patch_size
            if n_patches == 0:
                raise ValueError("La lunghezza della sequenza deve essere almeno pari a patch_size")
            # Tronca x per avere un numero intero di patch
            x = x[:, :n_patches * self.patch_size, :]
            # Riorganizza in patch: (B, n_patches, patch_size, input_size)
            x = x.reshape(B, n_patches, self.patch_size, D)
            # Appiattisce le patch: (B, n_patches, patch_size * input_size)
            x = x.reshape(B, n_patches, self.patch_size * D)
            # Proietta le patch in embedding: (B, n_patches, embed_dim)
            x = self.patch_projection(x)
        
        # Aggiunge il token speciale di regressione se abilitato
        if self.use_special_token:
            reg_tokens = self.reg_token.expand(B, -1, -1)
            x = torch.cat([x, reg_tokens], dim=1)
        
        # Aggiunge la codifica posizionale sinusoidale
        x = self.pos_encoding(x)
        
        # Elabora con l'encoder Transformer
        x = self.transformer(x)
        
        # Utilizza il token speciale per la previsione se abilitato, altrimenti usa l'ultimo token patch
        if self.use_special_token:
            x = self.fc(x[:, -1, :])  # Usa il token speciale (ultima posizione)
        else:
            x = self.fc(x[:, -1, :])  # Usa l'ultimo token patch
            
        return x
