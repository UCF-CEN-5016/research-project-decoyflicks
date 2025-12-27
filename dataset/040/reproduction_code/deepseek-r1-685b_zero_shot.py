import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).view(B, C, -1)
        k = self.k(x).view(B, C, -1)
        v = self.v(x).view(B, C, -1)
        attn = torch.einsum('bci,bcj->bij', q, k) * (C ** -0.5)
        attn = attn.softmax(dim=1)  # Potential bug: should be dim=2?
        out = torch.einsum('bij,bcj->bci', attn, v)
        out = out.view(B, C, H, W)
        return self.proj(out)

# Test case
x = torch.randn(2, 16, 8, 8)
attn = Attention(16)
out = attn(x)
print(out.shape)