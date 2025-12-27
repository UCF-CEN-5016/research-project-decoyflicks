import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.arange(half_dim, device=device, dtype=torch.float32)
        emb = self.theta ** (-2 * (emb // 2) / (self.dim // 2))
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

emb = SinusoidalPosEmb(dim=4)
x = torch.ones((10))
out = emb(x)
print(out, out.shape)