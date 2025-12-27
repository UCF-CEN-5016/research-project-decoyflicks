import torch
import torch.nn as nn

class QKNormAttention(nn.Module):
    def __init__(self, heads, kv_heads, dim_head):
        super(QKNormAttention, self).__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.dim_head = dim_head
        
        # Initialize qk_norm with correct dimensions
        self.qk_norm = nn.Parameter(torch.ones(kv_heads, 1, dim_head))

    def forward(self, q, k, v):
        # Simulate attention operation with shape matching
        q = q * self.qk_norm
        return q

# Create an instance of the model
model = QKNormAttention(heads=8, kv_heads=4, dim_head=64)
q = torch.randn(1, 8, 64)  # Query with 8 heads
k = torch.randn(1, 4, 64)  # Key with 4 heads

# Run the model forward pass
output = model(q, k, k)
print(output.shape)