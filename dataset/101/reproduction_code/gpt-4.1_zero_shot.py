import torch
from einops import rearrange

def rotate_queries_and_keys(Q, K, seq_len, rotary_dim):
    pos = torch.arange(seq_len, device=Q.device)
    freqs = 1.0 / (10000 ** (torch.arange(0, rotary_dim, 2, device=Q.device).float() / rotary_dim))
    freqs = pos[:, None] * freqs[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    emb = emb[None, :, None, :]

    def apply_rotary(x):
        x_rot = rearrange(x[..., :rotary_dim], 'b s h d -> b s h (d 1)')
        x1, x2 = x_rot[..., 0::2], x_rot[..., 1::2]
        cos_emb, sin_emb = emb[..., 0::2], emb[..., 1::2]
        x_rotated = torch.cat([x1 * cos_emb - x2 * sin_emb, x1 * sin_emb + x2 * cos_emb], dim=-1)
        return torch.cat([x_rotated.squeeze(-2), x[..., rotary_dim:]], dim=-1)

    Q_rot = apply_rotary(Q)
    K_rot = apply_rotary(K)
    return Q_rot, K_rot

torch.manual_seed(0)
b, s, h, d = 2, 16, 4, 32
rotary_dim = 20
Q = torch.randn(b, s, h, d, dtype=torch.float32)
K = torch.randn(b, s, h, d, dtype=torch.float32)
Q_rot, K_rot = rotate_queries_and_keys(Q, K, s, rotary_dim)
print(torch.isnan(K_rot).sum())