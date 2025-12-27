import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dim, dim_head=16, heads=8, kv_heads=4, qk_norm=True):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.qk_norm = qk_norm
        self.to_q = nn.Linear(dim, dim_head * heads)
        self.to_k = nn.Linear(dim, dim_head * kv_heads)
        self.to_v = nn.Linear(dim, dim_head * kv_heads)
        self.qk_norm_k_scale = nn.Parameter(torch.ones(kv_heads, 1, dim_head)) if qk_norm else None

    def forward(self, x, context=None):
        b, n, _ = x.shape
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = q.view(b, n, self.heads, -1).transpose(1, 2)
        k = k.view(b, n, self.kv_heads, -1).transpose(1, 2)
        v = v.view(b, n, self.kv_heads, -1).transpose(1, 2)

        if self.qk_norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)

        # Attention computation (simplified)
        attn = (q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        out = attn @ v
        return out

# Parameters
dim = 64
heads = 8
kv_heads = 4
dim_head = 16
batch_size = 2
seq_length = 10

# Input tensor
x = torch.rand(batch_size, seq_length, dim)

# Initialize Attention module
attention = Attention(dim=dim, heads=heads, kv_heads=kv_heads, qk_norm=True)

# Context tensor
context = torch.rand(batch_size, seq_length, dim)

# Forward pass
output = attention(x, context)

# Check output shape
print("Output shape:", output.shape)

# Check for NaN values
assert not torch.isnan(output).any(), "Output contains NaN values"

# Modify initialization
attention.qk_norm_k_scale = nn.Parameter(torch.ones(kv_heads, 1, dim_head))

# Re-run forward method
output_modified = attention(x, context)

# Check output shape again
print("Modified output shape:", output_modified.shape)

# Assert outputs are different
assert not torch.equal(output, output_modified), "Outputs are the same, bug not present"