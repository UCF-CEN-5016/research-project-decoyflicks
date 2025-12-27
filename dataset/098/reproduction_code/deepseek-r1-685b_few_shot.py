import torch
import math
from torch import nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim  # Missing self.theta assignment

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(theta) / (half_dim - 1)  # Will raise NameError
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# Test case that triggers the error
emb = SinusoidalPosEmb(dim=4)
x = torch.ones(10)
out = emb(x)  # Raises NameError: 'theta' is not defined