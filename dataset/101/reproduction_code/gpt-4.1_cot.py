import torch
import math

# Rotary embeddings implementation (simplified)
def rotary_emb(freqs, t):
    # freqs: [seq_len, dim//2]
    # t: [batch, seq_len, dim]
    # Apply rotary embedding to last dimension split in half
    t1, t2 = t[..., ::2], t[..., 1::2]  # even and odd dims
    cos = freqs.cos()
    sin = freqs.sin()
    # rotate dims:
    # (x, y) -> (x*cos - y*sin, x*sin + y*cos)
    rotated_even = t1 * cos - t2 * sin
    rotated_odd = t1 * sin + t2 * cos
    # interleave back
    out = torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)
    return out

# Generate rotary frequencies with xpos scaling (causes NaNs sometimes)
def get_rotary_freqs(dim, seq_len, base=10000, xpos_power=1.0):
    # dim: must be even
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len).float()
    # xpos: scale t with a power (this is the suspicious part)
    scaled_t = t ** xpos_power
    freqs = torch.einsum("i,j->ij", scaled_t, inv_freq)  # [seq_len, dim//2]
    return freqs

# Simulate Q, K tensors
batch_size = 2
seq_len = 128
dim = 64  # must be even

# Initialize Q, K with normal values
Q = torch.randn(batch_size, seq_len, dim)
K = torch.randn(batch_size, seq_len, dim)

# Create rotary frequencies with xpos_power > 1 to simulate weird scaling
freqs = get_rotary_freqs(dim, seq_len, xpos_power=2.5)  # raising t to a power >1 can explode values

# Apply rotary embedding to Q and K
Q_rot = rotary_emb(freqs, Q)
K_rot = rotary_emb(freqs, K)

# Check for NaNs
print("NaNs in Q_rot:", torch.isnan(Q_rot).sum().item())
print("NaNs in K_rot:", torch.isnan(K_rot).sum().item())