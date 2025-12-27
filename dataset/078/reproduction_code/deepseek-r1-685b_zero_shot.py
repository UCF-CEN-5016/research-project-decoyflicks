import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange

class NaViT3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_patch_embedding = nn.Linear(3, 1024)
        self.layers = nn.ModuleList([nn.LayerNorm(1024) for _ in range(2)])
        self.mlp_head = nn.Linear(1024, 10)

    def forward(self, x, mask):
        x = self.to_patch_embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)
        return self.mlp_head(x)

model = NaViT3D()
optimizer = torch.optim.Adam(model.parameters())

x = torch.randn(5, 1024, 3)
mask = torch.ones(5, 1024).bool()
x = torch.nested.nested_tensor([x[i][mask[i]] for i in range(5)])

out = model(x, mask)
loss = out.sum()
loss.backward()