import torch
import torch.nn as nn

class RegionViT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        return self.norm(x)

dim = 64
model = RegionViT(dim)
x = torch.randn(1, dim, 32, 32)
out = model(x)