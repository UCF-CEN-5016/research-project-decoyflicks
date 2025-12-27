import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d, base=10000):
        super().__init__()
        self.d = d
        self.base = base
        self.register_buffer('cos_cached', None, persistent=False)
        self.register_buffer('sin_cached', None, persistent=False)

    def _build_cache(self, x):
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return
        seq_len = x.shape[0]
        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d))
        seq_idx = torch.arange(seq_len).float()
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)
        self.register_buffer('cos_cached', idx_theta2.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', idx_theta2.sin()[None, None, :, :], persistent=False)

    def forward(self, x):
        self._build_cache(x)
        x_rope, x_pass = x[..., :self.d], x[..., self.d:]
        neg_half_x = torch.cat([-x_rope[..., self.d//2:], x_rope[..., :self.d//2]], dim=-1)
        x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])
        return torch.cat((x_rope, x_pass), dim=-1)

rope = RotaryPositionalEmbedding(d=4)
x = torch.randn(3, 1, 8)
output = rope(x)