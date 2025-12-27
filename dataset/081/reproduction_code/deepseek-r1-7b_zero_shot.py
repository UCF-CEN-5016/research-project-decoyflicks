import torch
import torch.nn as nn

class ImageEmbedder(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, kernel_size: int = 5):
        super().__init__()
        self.patch_embedding = nn.Conv2d(in_channels=in_channels,
                                          kernel_size=kernel_size,
                                          stride=kernel_size)
        # Calculate output features based on desired embedding dimension
        # This is a simplification; actual calculation may vary per layer's needs
        if embed_dim > 0:
            out_features = self.patch_embedding.out_channels * (embed_dim // self.patch_embedding.out_channels)
        else:
            out_features = self.patch_embedding.out_channels
        
        self.position_embedding = nn.Embedding(pos_dim, out_features)  # Adjusted to match output features

    def forward(self, x):
        x = self.patch_embedding(x)
        b, c, h, w = x.size()
        seq_len = h * w
        x = x.view(b, c, seq_len).transpose(1,2)
        x += self.position_embedding(torch.arange(seq_len, device=x.device))
        return x

class MultiScaleMaskedAutoencoder(nn.Module):
    def __init__(self, in_channels: int=3, patch_size: int=5):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            ImageEmbedder(in_channels=in_channels, embed_dim=192),
            nn.LayerNorm([192])
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(192, (in_channels // patch_size**2) * 3),  # Adjusted based on input channels and patch size
            nn.Unflatten(0, (in_channels // patch_size**2, 5, 5)),
            nn.PixelShuffle()
        )
        
        self.posterior = nn.Linear(192, in_channels // (patch_size**2))
    
    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x), self.posterior(x)

class CrossViT(nn.Module):
    def __init__(self, num_classes: int=1000, patch_size: int=5, in_channels: int=3,
                 d_model: int=512, nhead: int=8, dim_feedforward: int=2048, n encoder_layers: int=6):
        super().__init__()
        self.transformer = MultiScaleMaskedAutoencoder(
            in_channels=in_channels,
            patch_size=patch_size
        )