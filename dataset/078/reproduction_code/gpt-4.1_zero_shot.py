import torch
from torch import nn
from torch.nn import functional as F

class DummyNavitNest3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Parameter(torch.randn(5, 2, 1024))
        self.k = nn.Parameter(torch.randn(5, 3, 1024))
        self.v = nn.Parameter(torch.randn(5, 3, 1024))

    def forward(self):
        q = self.q
        k = self.k
        v = self.v
        attn = torch.einsum('bjd,bkd->bjk', q, k) / (1024 ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bjk,bkd->bjd', attn, v)
        return out.sum()

model = DummyNavitNest3D()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for _ in range(3):
    opt.zero_grad()
    loss = model()
    loss.backward()
    opt.step()