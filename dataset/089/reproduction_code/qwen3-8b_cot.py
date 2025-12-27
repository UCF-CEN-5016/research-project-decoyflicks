import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, dim, heads=4, kv_heads=2, dim_head=64):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.dim_head = dim_head
        
        # Simulate the buggy line from the original code
        # This line causes a shape mismatch when kv_heads != heads
        self.qk_norm_k_scale = nn.Parameter(torch.ones(heads, 1, dim_head))
        
        # Other components (simplified)
        self.to_q = nn.Linear(dim, heads * dim_head, bias=False)
        self.to_k = nn.Linear(dim, kv_heads * dim_head, bias=False)
        self.to_v = nn.Linear(dim, kv_heads * dim_head, bias=False)
        self.to_out = nn.Linear(kv_heads * dim_head, dim, bias=False)

    def forward(self, x):
        # Simulate forward pass
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        
        # Reshape for attention
        q = q.view(x.shape[0], -1, self.heads, self.dim_head)
        k = k.view(x.shape[0], -1, self.kv_heads, self.dim_head)
        v = v.view(x.shape[0], -1, self.kv_heads, self.dim_head)
        
        # Apply qk_norm_k_scale (this will fail due to shape mismatch)
        k = k * self.qk_norm_k_scale
        
        # Dummy attention calculation
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.dim_head ** 0.5)
        out = torch.matmul(attn, v)
        out = out.reshape(x.shape[0], -1, self.dim)
        return self.to_out(out)

# Reproduce the bug
dim = 128
model = Attention(dim, heads=4, kv_heads=2, dim_head=64)
input_tensor = torch.randn(1, 16, dim)  # Batch size 1, sequence length 16

# Trigger the bug
try:
    output = model(input_tensor)
    print("No error occurred. The bug was not reproduced.")
except Exception as e:
    print(f"Bug reproduced: {e}")