import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * (self.dim ** -0.5)
        return self.scale * x / (norm + self.eps)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.,
        kv_heads=None,  # Number of key-value heads
        qk_norm=False   # Enable query-key normalization
    ):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads or heads  # If None, use same as heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        inner_dim_kv = dim_head * self.kv_heads

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim_kv, bias=False)
        self.to_v = nn.Linear(dim, inner_dim_kv, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Query-Key normalization
        self.qk_norm = qk_norm
        if qk_norm:
            # BUG: This should use self.kv_heads instead of heads when kv_heads != heads
            # Incorrect parameter shape when kv_heads != heads
            self.qk_norm_q_scale = nn.Parameter(torch.ones(heads, 1, dim_head))
            self.qk_norm_k_scale = nn.Parameter(torch.ones(heads, 1, dim_head))  # BUG: Should be kv_heads
            
            # Correct implementation would be:
            # self.qk_norm_q_scale = nn.Parameter(torch.ones(heads, 1, dim_head))
            # self.qk_norm_k_scale = nn.Parameter(torch.ones(self.kv_heads, 1, dim_head))

    def forward(self, x, context=None, mask=None):
        b, n, _, h, kv_h = *x.shape, self.heads, self.kv_heads
        
        context = context if context is not None else x
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=kv_h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=kv_h)
        
        # Apply QK normalization if enabled
        if self.qk_norm:
            # Normalize query
            q = F.normalize(q, dim=-1)
            q = q * self.qk_norm_q_scale
            
            # Normalize key
            k = F.normalize(k, dim=-1)
            try:
                # This will fail when kv_heads != heads due to shape mismatch
                k = k * self.qk_norm_k_scale
            except RuntimeError as e:
                print(f"Error in QK normalization: {e}")
                print(f"qk_norm_k_scale shape: {self.qk_norm_k_scale.shape}, k shape: {k.shape}")
                raise
        
        # Handle multi-head attention with different number of KV heads
        if kv_h != h:
            k = repeat(k, 'b h n d -> b (h r) n d', r=h//kv_h)
            v = repeat(v, 'b h n d -> b (h r) n d', r=h//kv_h)
        
        # Compute attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        if mask is not None:
            dots = dots.masked_fill(mask, -1e9)
        
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        return out

def test_qk_norm_kv_heads_conflict():
    # Test with different combinations of qk_norm and kv_heads
    dim = 256
    heads = 8
    
    # Test 1: qk_norm=True, kv_heads=heads (should work)
    print("\nTest 1: qk_norm=True, kv_heads=heads")
    try:
        attn1 = Attention(dim=dim, heads=heads, qk_norm=True, kv_heads=heads)
        x1 = torch.randn(2, 16, dim)
        out1 = attn1(x1)
        print("✓ Success: qk_norm=True, kv_heads=heads")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 2: qk_norm=False, kv_heads!=heads (should work)
    print("\nTest 2: qk_norm=False, kv_heads!=heads")
    try:
        attn2 = Attention(dim=dim, heads=heads, qk_norm=False, kv_heads=2)
        x2 = torch.randn(2, 16, dim)
        out2 = attn2(x2)
        print("✓ Success: qk_norm=False, kv_heads!=heads")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 3: qk_norm=True, kv_heads!=heads (should fail due to bug)
    print("\nTest 3: qk_norm=True, kv_heads!=heads (reproduces the bug)")
    try:
        attn3 = Attention(dim=dim, heads=heads, qk_norm=True, kv_heads=2)
        x3 = torch.randn(2, 16, dim)
        out3 = attn3(x3)
        print("✓ Unexpectedly succeeded: qk_norm=True, kv_heads!=heads")
    except Exception as e:
        print(f"✗ Failed as expected (bug reproduced): {e}")

if __name__ == "__main__":
    test_qk_norm_kv_heads_conflict()