import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dim, dim_head, heads, kv_heads, qk_norm, qk_norm_groups):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.qk_norm = qk_norm
        self.qk_norm_groups = qk_norm_groups
        self.to_q = nn.Linear(dim, dim_head * heads)
        self.to_k = nn.Linear(dim, dim_head * kv_heads)
        self.to_v = nn.Linear(dim, dim_head * kv_heads)

    def forward(self, x, context):
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        return q, k, v

dim = 64
heads = 8
kv_heads = 4
dim_head = 16
batch_size = 2
seq_length = 10

input_tensor = torch.rand(batch_size, seq_length, dim)
context_tensor = torch.rand(batch_size, seq_length, kv_heads * dim_head)

attention = Attention(dim=dim, dim_head=dim_head, heads=heads, kv_heads=kv_heads, qk_norm=True, qk_norm_groups=2)
output = attention(input_tensor, context_tensor)

assert output[0].shape == (batch_size, seq_length, heads * dim_head)
assert not torch.isnan(output[0]).any()

print("Input shape:", input_tensor.shape)
print("Output shape:", output[0].shape)