import torch
import torch.nn as nn
from vit_pytorch.regionvit import RegionViT

# Create a RegionViT model
regionvit = RegionViT(
    dim = 64,
    depth = 6,
    heads = 8,
    window_size = 7,
    dim_head = 32
)

# Create test input
x = torch.randn(2, 224, 224, 3)

# Extract the problematic layer norm
original_norm = regionvit.local_transformer.layers[0].norm1

# Create correct implementation for comparison
correct_norm = nn.LayerNorm(64)  # Assuming dim=64 should be normalized

# Test embeddings
embeddings = torch.randn(2, 56, 56, 64)  # Channel-last format

# Apply both normalizations
original_output = original_norm(embeddings)
correct_output = correct_norm(embeddings)  # Will normalize over last dimension

# Compare results
diff = (original_output - correct_output).abs().mean()
print(f"Mean difference between implementations: {diff}")
print(f"Original shape: {original_output.shape}, Correct shape: {correct_output.shape}")