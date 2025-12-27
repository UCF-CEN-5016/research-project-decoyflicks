import torch
from torch import nn

class XTransformer(nn.Module):
    def __init__(self, dim_head=32, kv_heads=2):
        super().__init__()
        
        # Demonstrate the bug with conflicting configurations
        self.qk_norm_k_scale = nn.Parameter(torch.ones(kv_heads, 1, dim_head))  # Fixed vs Learnable
        
# Example usage where kv_heads (2) != qk_norm (3)
kv_heads = 2
qk_norm = 3

# Create an instance to demonstrate the bug and fix
model = XTransformer(dim_head=qk_norm)