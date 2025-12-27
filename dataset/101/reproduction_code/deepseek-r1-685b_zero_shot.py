import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, pos, t):
        return (t * pos.cos()) + (self.rotate_half(t) * pos.sin())

    def rotate_queries_and_keys(self, q, k):
        seq_len = q.shape[-2]
        pos = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", pos, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return self.apply_rotary_pos_emb(emb[None, :, :], q), self.apply_rotary_pos_emb(emb[None, :, :], k)

def test_rotary_nan():
    dim = 64
    seq_len = 128
    batch_size = 8
    heads = 4
    rotary = RotaryEmbedding(dim)
    Q = torch.randn(batch_size, heads, seq_len, dim)
    K = torch.randn(batch_size, heads, seq_len, dim)
    Q_rot, K_rot = rotary.rotate_queries_and_keys(Q, K)
    print(f"NaNs in K: {torch.isnan(K_rot).any().item()}")

test_rotary_nan()