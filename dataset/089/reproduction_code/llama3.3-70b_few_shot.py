import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, dim, heads, kv_heads, dim_head):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.kv_heads = kv_heads
        self.dim_head = dim_head
        self.qk_norm_k_scale = nn.Parameter(torch.ones(heads, 1, dim_head))  # Incorrect initialization

    def forward(self, x):
        # Simulate qk_norm and kv_heads conflict
        q = x[:, :self.heads, :]
        k = x[:, :self.kv_heads, :]
        return self.qk_norm_k_scale * k

# Initialize transformer with conflicting configuration
transformer = Transformer(dim=128, heads=8, kv_heads=16, dim_head=32)

# Input data
x = torch.randn(1, 16, 128)

# This will cause a dimension mismatch error
try:
    output = transformer(x)
    print("Output shape:", output.shape)
except RuntimeError as e:
    print("Error:", e)