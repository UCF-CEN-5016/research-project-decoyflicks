import torch
import torch.nn as nn

class MultiHeadLinear(nn.Module):
    def __init__(self, in_features, out_features, num_heads):
        super(MultiHeadLinear, self).__init__()
        self.heads = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(num_heads)])

    def forward(self, x):
        return torch.stack([head(x) for head in self.heads], dim=-1)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, kv_heads=2, dim_head=64):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.dim_head = dim_head

        self.q_linear = MultiHeadLinear(dim, dim_head, heads)
        self.k_linear = MultiHeadLinear(dim, dim_head, kv_heads)
        self.v_linear = MultiHeadLinear(dim, dim_head, kv_heads)
        self.to_out = nn.Linear(kv_heads * dim_head, dim, bias=False)

    def forward(self, x):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        q = q.view(x.shape[0], -1, self.heads, self.dim_head)
        k = k.view(x.shape[0], -1, self.kv_heads, self.dim_head)
        v = v.view(x.shape[0], -1, self.kv_heads, self.dim_head)

        k = k * q.mean(dim=2, keepdim=True)  # Apply normalization by mean of q

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