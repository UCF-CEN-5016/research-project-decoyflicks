import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., qk_norm=False, kv_heads=None):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads if kv_heads is not None else heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        inner_dim_kv = dim_head * self.kv_heads
        
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim_kv, bias=False)
        self.to_v = nn.Linear(dim, inner_dim_kv, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        
        # QK normalization parameters - bug occurs here
        self.qk_norm = qk_norm
        if qk_norm:
            # This is the bug - using heads instead of kv_heads for k_scale
            self.qk_norm_q_scale = nn.Parameter(torch.ones(heads, 1, dim_head))
            self.qk_norm_k_scale = nn.Parameter(torch.ones(heads, 1, dim_head))
            # Correct version would be:
            # self.qk_norm_k_scale = nn.Parameter(torch.ones(self.kv_heads, 1, dim_head))

    def forward(self, x):
        b, n, _, h, kv_h = *x.shape, self.heads, self.kv_heads
        
        # Project to q, k, v
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        
        # Reshape for multi-head attention
        q = q.reshape(b, n, h, -1).transpose(1, 2)
        k = k.reshape(b, n, kv_h, -1).transpose(1, 2)
        v = v.reshape(b, n, kv_h, -1).transpose(1, 2)
        
        # Apply QK normalization - this will fail when kv_heads != heads
        if self.qk_norm:
            q = F.normalize(q, dim=-1)
            q = q * self.qk_norm_q_scale
            
            k = F.normalize(k, dim=-1)
            # This will cause a shape mismatch when kv_heads != heads
            k = k * self.qk_norm_k_scale
        
        return q, k, v

def test_bug():
    # Test case that reproduces the bug
    dim = 64
    heads = 8
    kv_heads = 2  # Different from heads
    qk_norm = True
    
    try:
        model = Attention(dim=dim, heads=heads, kv_heads=kv_heads, qk_norm=qk_norm)
        x = torch.randn(1, 16, dim)
        q, k, v = model(x)
        print("Test unexpectedly passed. This means the bug wasn't reproduced.")
    except Exception as e:
        print(f"Bug reproduced! Error: {e}")
        print(f"The error occurred because qk_norm_k_scale has shape for {heads} heads")
        print(f"but k has shape for {kv_heads} heads.")

if __name__ == "__main__":
    test_bug()