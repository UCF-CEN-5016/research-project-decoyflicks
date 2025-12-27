import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding module producing [sin, cos] pairs."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def _frequencies(self, device: torch.device) -> torch.Tensor:
        half_dim = self.dim // 2
        exponent = math.log(self.theta) / (half_dim - 1)
        return torch.exp(torch.arange(half_dim, device=device) * -exponent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        pos = x.view(-1)
        freqs = self._frequencies(device)
        angles = pos[:, None] * freqs[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


# Test script that triggers the NameError
emb = SinusoidalPosEmb(dim=4)
x = torch.ones((10))
out = emb(x)
print(out, out.shape)