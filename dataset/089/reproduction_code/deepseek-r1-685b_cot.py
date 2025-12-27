import torch
from torch import nn, einsum
from einops import rearrange

class Attention(nn.Module):
    def __init__(self, dim, heads=8, kv_heads=8, dim_head=64, qk_norm=False):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.scale = dim_head ** -0.5
        
        self.to_q = nn.Linear(dim, heads * dim_head, bias=False)
        self.to_kv = nn.Linear(dim, 2 * kv_heads * dim_head, bias=False)
        
        self.qk_norm = qk_norm
        if qk_norm:
            # Original buggy version:
            self.qk_norm_k_scale = nn.Parameter(torch.ones(heads, 1, dim_head))
            # Should be:
            # self.qk_norm_k_scale = nn.Parameter(torch.ones(kv_heads, 1, dim_head))
            
    def forward(self, x):
        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim=-1)
        
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.kv_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.kv_heads)
        
        if self.qk_norm:
            # This will fail when heads != kv_heads
            k = k * self.qk_norm_k_scale
        
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out

# Trigger the bug:
dim = 512
heads = 8
kv_heads = 4  # Different from heads
attn = Attention(dim, heads=heads, kv_heads=kv_heads, qk_norm=True)
x = torch.randn(1, 16, dim)  # (batch, seq, dim)

# This will raise shape mismatch error
try:
    out = attn(x)
except RuntimeError as e:
    print("Error occurred:", e)