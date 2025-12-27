import torch
from torch import nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def rotate_queries_and_keys(self, q, k):
        seq_len = q.shape[-2]
        t = torch.arange(seq_len, device=q.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        q_rot, k_rot = rotate_half(q, sin, cos), rotate_half(k, sin, cos)
        return q_rot, k_rot

def rotate_half(x, sin, cos):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)

# Setup
dim = 64
rotary_emb = RotaryEmbedding(dim)
batch_size = 2
seq_len = 128
heads = 8

# Simulate Q, K tensors
Q = torch.randn(batch_size, heads, seq_len, dim)
K = torch.randn(batch_size, heads, seq_len, dim)

# Rotate Q and K
Q_rot, K_rot = rotary_emb.rotate_queries_and_keys(Q, K)

# Check for NaNs
print("NaN count in K:", torch.isnan(K_rot).sum().item())  # Should be 0 but shows some NaNs
print("NaN positions:", torch.nonzero(torch.isnan(K_rot)))  # Shows where NaNs appear