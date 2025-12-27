import torch
from torch import nn

class CrossViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(16, 16, bias=False)
        self.to_k = nn.Linear(16, 16, bias=False)
        self.to_v = nn.Linear(16, 16, bias=False)
        self.to_out = nn.Linear(16, 16)

    def forward(self, x_large, x_small):
        q = self.to_q(x_large)
        k = self.to_k(x_small)
        v = self.to_v(x_small)
        dots = torch.einsum('b i d, b j d -> b i j', q, k)
        attn = dots.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        return self.to_out(out)

model = CrossViT()
x_large = torch.randn(2, 10, 16)
x_small = torch.randn(2, 5, 16)
model(x_large, x_small)