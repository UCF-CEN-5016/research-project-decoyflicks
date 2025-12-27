import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, dim, heads=8, kv_heads=4, dim_head=64, qk_norm=True):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.scale = dim_head ** -0.5
        
        if qk_norm:
            self.qk_norm_k_scale = nn.Parameter(torch.ones(heads, 1, dim_head))
            self.qk_norm_q_scale = nn.Parameter(torch.ones(heads, 1, dim_head))
        
    def forward(self, q, k, v):
        if hasattr(self, 'qk_norm_k_scale'):
            k = k * self.qk_norm_k_scale
            q = q * self.qk_norm_q_scale
        return torch.randn(q.shape[0], q.shape[1], q.shape[2])

dim = 512
heads = 8
kv_heads = 4
dim_head = 64
model = Attention(dim, heads, kv_heads, dim_head, qk_norm=True)
q = torch.randn(1, 10, heads * dim_head)
k = torch.randn(1, 10, kv_heads * dim_head)
v = torch.randn(1, 10, kv_heads * dim_head)
output = model(q, k, v)