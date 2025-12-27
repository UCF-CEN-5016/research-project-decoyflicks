import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.q = nn.Conv2d(n_channels, n_channels // 8, 1)
        self.k = nn.Conv2d(n_channels, n_channels // 8, 1)
        self.v = nn.Conv2d(n_channels, n_channels, 1)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.q(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.k(x).view(B, -1, H * W)
        v = self.v(x).view(B, -1, H * W)
        attn = torch.einsum('bji,bjk->bij', q, k)
        attn = attn.softmax(dim=1)
        out = torch.einsum('bij,bjk->bji', attn, v).permute(0, 2, 1).view(B, C, H, W)
        return self.gamma * out + x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.att = Attention(12)

    def forward(self, x):
        return self.att(x)

# setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
x = torch.randn(1, 12, 8, 8).to(device)

# reproduce
output = model(x)