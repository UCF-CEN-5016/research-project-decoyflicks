import torch
from vit_pytorch.mae import MAE

mae = MAE(
    encoder = torch.nn.Module(), 
    decoder_dim = 512,
    masking_ratio = 0.75
)

# Simulate the bug - missing colon in slice operation
num_patches = 196  # Example value
tokens = torch.randn(1, num_patches, 768)  # Example tokens

# This line contains the exact syntax error from the bug report
tokens = tokens + mae.encoder.pos_embedding[:, 1:(num_patches + 1)]