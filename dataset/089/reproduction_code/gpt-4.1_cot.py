import torch
from torch import nn
from x_transformers.x_transformers import XTransformer

# Minimal setup parameters
dim = 64
depth = 1
heads = 4
kv_heads = 2  # Different from heads to trigger the bug
qk_norm = True  # Enable qk_norm to trigger the conflict

# Create a minimal XTransformer model that uses qk_norm and kv_heads != heads
model = XTransformer(
    dim=dim,
    depth=depth,
    heads=heads,
    kv_heads=kv_heads,
    qk_norm=qk_norm,
    # other parameters can be default
)

# Create dummy input: (batch, seq_len, dim)
batch_size = 2
seq_len = 8
x = torch.randn(batch_size, seq_len, dim)

# Forward pass to trigger potential bug
out = model(x)

print("Output shape:", out.shape)