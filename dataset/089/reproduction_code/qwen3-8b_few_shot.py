import torch
import torch.nn as nn

class QKNormAttention(nn.Module):
    def __init__(self, heads, kv_heads, dim_head):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.dim_head = dim_head
        
        # Incorrectly initialized with `heads` instead of `kv_heads`
        self.qk_norm = nn.Parameter(torch.ones(heads, 1, dim_head))  # <-- Bug here

    def forward(self, q, k, v):
        # Simulate attention operation with shape mismatch
        q = q * self.qk_norm  # <-- Shape mismatch occurs here
        return q  # Simplified output

# Reproduce the conflict
model = QKNormAttention(heads=8, kv_heads=4, dim_head=64)
q = torch.randn(1, 8, 64)  # Query with 8 heads
k = torch.randn(1, 4, 64)  # Key with 4 heads

# This will raise a RuntimeError due to shape mismatch
output = model(q, k, k)
print(output.shape)