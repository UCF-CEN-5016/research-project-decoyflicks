import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=4, kv_heads=2, dim_head=64):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.dim_head = dim_head
        self.qk_norm = nn.LayerNorm(dim)
        self.qk_norm_k_scale = nn.Parameter(torch.ones(heads, 1, dim_head))

    def forward(self, x):
        key = torch.randn(1, 10, dim)
        key = key.view(1, 10, self.heads, self.dim_head)
        scaled_key = key * self.qk_norm_k_scale
        return scaled_key

dim = 4 * 64
model = AttentionBlock(dim, heads=4, kv_heads=2, dim_head=64)
model(torch.randn(1, 10, dim))