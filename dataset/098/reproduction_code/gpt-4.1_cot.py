import math
import torch
import torch.nn as nn

# Buggy version of SinusoidalPosEmb (theta not assigned in __init__)
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        # self.theta = theta  # <-- missing on purpose to reproduce the bug

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        # This line will raise NameError because theta is not defined in forward or as self.theta
        emb = math.log(theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# Minimal test to trigger the bug
if __name__ == "__main__":
    emb = SinusoidalPosEmb(dim=4)
    x = torch.ones((10,))
    out = emb(x)  # This should raise NameError: name 'theta' is not defined