import math
import torch
import torch.nn as nn
from typing import Optional


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding module.

    Produces a (N, dim) tensor of sinusoidal embeddings for a 1-D input tensor x of shape (N,).
    The first half of each embedding contains sine terms, the second half contains cosine terms.
    """
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def _inverse_frequencies(self, device: Optional[torch.device]) -> torch.Tensor:
        half_dim = self.dim // 2
        # same computation as original: log(theta) / (half_dim - 1)
        log_term = math.log(self.theta) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -log_term)
        return freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        freqs = self._inverse_frequencies(device)
        # outer product between x (N,) and freqs (half_dim,) => (N, half_dim)
        proj = x[:, None] * freqs[None, :]
        emb = torch.cat((proj.sin(), proj.cos()), dim=-1)
        return emb


if __name__ == "__main__":
    emb = SinusoidalPosEmb(dim=4)
    x = torch.ones((10,))
    out = emb(x)
    print(out, out.shape)