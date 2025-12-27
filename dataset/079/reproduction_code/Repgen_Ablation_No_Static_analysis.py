import torch
import torch.nn as nn

# Create a standard PyTorch LayerNorm
layer_norm = nn.LayerNorm(64)

# Create test tensor (batch_size, height, width, channels)
x = torch.randn(2, 56, 56, 64)

# Apply layer norm (will correctly normalize over last dimension)
normalized = layer_norm(x)

# Verify normalization occurred correctly
mean = normalized.mean(dim=-1)
var = normalized.var(dim=-1, unbiased=False)

print(f"Mean close to 0: {torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)}")
print(f"Variance close to 1: {torch.allclose(var, torch.ones_like(var), atol=1e-5)}")