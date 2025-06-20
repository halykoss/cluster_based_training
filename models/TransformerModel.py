import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    """
    PatchTST model for time series forecasting:
      - Splits input time series into patches of fixed length.
      - Each patch is flattened and projected into an embedding space.
      - A learnable positional embedding is added to the patch embeddings.
      - A Transformer encoder processes the patch tokens.
      - The final patch's representation is used for prediction.
    
    Args:
        input_size (int): Number of features in the input time series.
        output_size (int): The size of the output.
        patch_size (int, optional): Length of each patch. Default is 16.
        embed_dim (int, optional): Dimension of patch embeddings. Default is 64.
        num_layers (int, optional): Number of Transformer encoder layers. Default is 2.
        num_heads (int, optional): Number of attention heads in each Transformer layer. Default is 4.
        use_cnn (bool, optional): Whether to use CNN for patch projection instead of linear layer. Default is False.
        use_special_token (bool, optional): Whether to use a special regression token for prediction. Default is False.
    """
    def __init__(self, input_size, output_size, patch_size=16, embed_dim=64, num_layers=2, 
                num_heads=4, use_cnn=False, use_special_token=False):
        super(TransformerModel, self).__init__()
        self.patch_size = patch_size
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.use_cnn = use_cnn
        self.use_special_token = use_special_token
        
        # Projection module: either CNN or Linear
        if use_cnn:
            # CNN projection: kernel_size = patch_size, stride = patch_size
            self.patch_projection = nn.Conv1d(
                in_channels=input_size,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size
            )
        else:
            # Linear projection for flattened patches (patch_size * input_size -> embed_dim)
            self.patch_projection = nn.Linear(patch_size * input_size, embed_dim)
        
        # Special regression token if required
        if use_special_token:
            self.reg_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Learnable positional embedding; max number of patches set arbitrarily (e.g., 500)
        # Add +1 to accommodate the regression token if used
        max_seq_len = 501 if use_special_token else 500
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        
        # Transformer encoder with batch_first=True.
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Fully connected output layer
        self.fc = nn.Linear(embed_dim, output_size)
    
    def forward(self, x):
        # x: (batch, sequence_length, input_size)
        B, L, D = x.shape
        
        if self.use_cnn:
            # Reshape to (B, D, L) for CNN
            x = x.transpose(1, 2)
            # Apply CNN projection
            x = self.patch_projection(x)  # Output: (B, embed_dim, n_patches)
            # Reshape to (B, n_patches, embed_dim)
            x = x.transpose(1, 2)
            n_patches = x.shape[1]
        else:
            n_patches = L // self.patch_size
            if n_patches == 0:
                raise ValueError("Sequence length must be at least as long as patch_size")
            # Crop x to have an integer number of patches
            x = x[:, :n_patches * self.patch_size, :]
            # Reshape into patches: (B, n_patches, patch_size, input_size)
            x = x.reshape(B, n_patches, self.patch_size, D)
            # Flatten patches: (B, n_patches, patch_size * input_size)
            x = x.reshape(B, n_patches, self.patch_size * D)
            # Project patches to embeddings: (B, n_patches, embed_dim)
            x = self.patch_projection(x)
        
        # Add special regression token if enabled
        if self.use_special_token:
            reg_tokens = self.reg_token.expand(B, -1, -1)
            x = torch.cat([x, reg_tokens], dim=1)
            pos_len = n_patches + 1
        else:
            pos_len = n_patches
        
        # Add positional embeddings
        x = x + self.pos_embedding[:, :pos_len, :]
        
        # Process with Transformer encoder
        x = self.transformer(x)
        
        # Use the special token for prediction if enabled, otherwise use the final patch
        if self.use_special_token:
            x = self.fc(x[:, -1, :])  # Use the special token (last position)
        else:
            x = self.fc(x[:, -1, :])  # Use the final patch token
        
        return x