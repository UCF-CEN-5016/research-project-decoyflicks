import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants
heads = 8
kv_heads = 4
dim_head = 64
batch_size = 2

# Input tensor
input_tensor = torch.randn(batch_size, heads * dim_head)

# Model initialization
class AttentionLayers(nn.Module):
    def __init__(self, heads, kv_heads, qk_norm):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.qk_norm = qk_norm

    def forward(self, x):
        # Simulated attention mechanism
        return x

model = AttentionLayers(heads=heads, kv_heads=kv_heads, qk_norm=True)

# Forward pass
output = model(input_tensor)

# Check for NaN values
if torch.isnan(output).any():
    print("Bug reproduced: NaN values found in output.")