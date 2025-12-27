import torch
import torch.nn.functional as F
import math

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=100):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.inv_freq = 1 / (10000 ** (torch.arange(0, dim, 2) / dim))

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]

        t = torch.arange(seq_len, device=x.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)

        if x.ndim == 3:
            return self.rotate(x, emb)
        else:
            return emb

    def rotate(self, x, emb):
        return (x * emb.cos()) + (x[..., 1::2] * emb.sin()).rotate(-1, dims=-1)


# Rotary embedding setup
rotary_emb = RotaryEmbedding(dim=128)

# Sample Q and K matrices
Q = torch.randn(1, 10, 128)
K = torch.randn(1, 10, 128)

# Rotate Q and K using rotary embedding
xpos = torch.arange(10)
rotated_Q = rotary_emb.rotate(Q, rotary_emb(xpos))
rotated_K = rotary_emb.rotate(K, rotary_emb(xpos))

# Check for NaN values
print("NaNs in rotated_Q:", torch.isnan(rotated_Q).sum())
print("NaNs in rotated_K:", torch.isnan(rotated_K).sum())