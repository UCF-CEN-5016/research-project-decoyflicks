import torch
from vit_pytorch.regionvit import RegionViT

# Create a RegionViT model
model = RegionViT(
    dim = 64,
    depth = 6,
    heads = 8,
    window_size = 7,
    dim_head = 32,
    attn_dropout = 0.1,
    ff_dropout = 0.1
)

# Create input with channel-last format (B, H, W, C)
x = torch.randn(2, 224, 224, 3)

# Forward pass - this will use the incorrect LayerNorm dimensions
output = model(x)

# Check normalized embeddings in specific layer
layer_norm = model.local_transformer.layers[0].norm1
embeddings = torch.randn(2, 64, 56, 56)  # Simulated embeddings
normalized = layer_norm(embeddings)

# Verify the issue: normalization should happen over channels (dim=-1)
# but might be happening over incorrect dimensions
print(f"Input shape: {embeddings.shape}, Output shape: {normalized.shape}")