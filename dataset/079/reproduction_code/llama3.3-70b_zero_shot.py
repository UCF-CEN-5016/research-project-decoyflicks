import torch
import torch.nn as nn

class RegionViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(128)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x

model = RegionViT()
x = torch.randn(1, 128, 16, 16)
print(model(x).shape)