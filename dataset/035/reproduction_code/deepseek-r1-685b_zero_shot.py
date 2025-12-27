import torch
import torch.nn as nn
import math

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d: int, base: int = 10_000):
        super().__init__()
        self.base = base
        self.d = d
        self.freqs = None

    def _compute_freqs(self, seq_len: int, device: torch.device):
        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2, dtype=torch.float, device=device) / self.d))
        seq = torch.arange(seq_len, device=device, dtype=torch.float)
        freqs = torch.einsum("..., l->...l", theta, seq)
        freqs = torch.cat([freqs, freqs], dim=-1)
        return freqs

    def forward(self, x: torch.Tensor):
        seq_len = x.shape[-2]
        if self.freqs is None or seq_len > self.freqs.shape[-1]:
            self.freqs = self._compute_freqs(seq_len, x.device)
        return x * self.freqs.cos() + self._rotate_half(x) * self.freqs.sin()

    def _rotate_half(self, x: torch.Tensor):
        d = x.shape[-1]
        x1, x2 = x[..., :d//2], x[..., d//2:]
        return torch.cat((-x2, x1), dim=-1)

x = torch.randn(1, 10, 4)
rotary_pe = RotaryPositionalEmbeddings(3)
rotary_pe(x)