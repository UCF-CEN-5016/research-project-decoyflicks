import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        # Missing: self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(theta) / (half_dim - 1)  # theta is not defined here
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# Instantiate and run
emb = SinusoidalPosEmb(dim=4)
x = torch.ones(10)
out = emb(x)  # Raises NameError: name 'theta' is not defined
print(out)