import torch
import math
from torch import nn

# Buggy implementation
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(theta) / (half_dim - 1)  # This will raise NameError
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# Test code that triggers the bug
if __name__ == '__main__':
    emb = SinusoidalPosEmb(dim=4)
    x = torch.ones((10))
    out = emb(x)  # This will raise NameError