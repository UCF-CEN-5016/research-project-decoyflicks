import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """Attention module with QK normalization and KV heads support"""
    
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.,
        qk_norm=False,
        kv_heads=None
    ):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads if kv_heads is not None else heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        inner_dim_kv = dim_head * self.kv_heads
        
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim_kv, bias=False)
        self.to_v = nn.Linear(dim, inner_dim_kv, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # QK normalization
        self.qk_norm = qk_norm
        if qk_norm:
            # Bug reproduction: Using heads for both q and k scale
            self.qk_norm_q_scale = nn.Parameter(torch.ones(heads, 1, dim_head))
            self.qk_norm_k_scale = nn.Parameter(torch.ones(heads, 1, dim_head))
            
            # Fixed version would use:
            # self.qk_norm_k_scale = nn.Parameter(torch.ones(self.kv_heads, 1, dim_head))

    def forward(self, x):
        b, n, d = x.shape
        h, kv_h = self.heads, self.kv_heads
        
        # Project input to queries, keys, and values
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        
        # Reshape for multi-head attention
        q = q.view(b, n, h, self.dim_head).transpose(1, 2)
        k = k.view(b, n, kv_h, self.dim_head).transpose(1, 2)
        v = v.view(b, n, kv_h, self.dim_head).transpose(1, 2)
        
        # Apply QK normalization
        if self.qk_norm:
            q = F.normalize(q, dim=-1)
            q = q * self.qk_norm_q_scale
            
            k = F.normalize(k, dim=-1)
            try:
                # This will fail when kv_heads != heads
                k = k * self.qk_norm_k_scale
                print(f"Applied k_scale with shape {self.qk_norm_k_scale.shape} to k with shape {k.shape}")
            except Exception as e:
                print(f"Error applying k_scale: {e}")
                print(f"k shape: {k.shape}, k_scale shape: {self.qk_norm_k_scale.shape}")
                
                # Try to fix the issue by creating a properly sized k_scale
                print("Attempting to fix the issue...")
                fixed_k_scale = self.qk_norm_k_scale[:kv_h] if kv_h < h else self.qk_norm_k_scale.repeat(kv_h // h + 1, 1, 1)[:kv_h]
                k = k * fixed_k_scale
                print(f"Applied fixed k_scale with shape {fixed_k_scale.shape} to k")
        
        # Handle different number of KV heads if needed
        if kv_h != h:
            k = k.repeat_interleave(h // kv_h, dim=0) if h > kv_h else k[:h]
            v = v.repeat_interleave(h // kv_h, dim=0) if h > kv_h else v[:h]
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * (self.dim_head ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

def test_bug_and_fix():
    """Test both the bug and a potential fix"""
    
    # Test parameters
    dim = 128
    heads_values = [8, 8, 8]
    kv_heads_values = [8, 2, 4]
    qk_norm_values = [True, True, True]
    
    print("Testing different configurations of qk_norm and kv_heads:")
    
    for i, (heads, kv_heads, qk_norm) in enumerate(zip(heads_values, kv_heads_values, qk_norm_values)):
        print(f"\nTest {i+1}: heads={heads}, kv_heads={kv_heads}, qk_norm={qk_norm}")
        
        try:
            model = Attention(
                dim=dim,
                heads=heads,
                kv_heads=kv_heads,
                qk_norm=qk_norm
            )
            
            # Run a forward pass
            x = torch.randn(2, 16, dim)
            output = model(x)
            
            print(f"✓ Test {i+1} passed! Output shape: {output.shape}")
        except Exception as e:
            print(f"✗ Test {i+1} failed: {e}")

if __name__ == "__main__":
    test_bug_and_fix()