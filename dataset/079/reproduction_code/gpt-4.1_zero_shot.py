import torch
from torch import nn

x = torch.randn(1, 64, 8, 8)
ln = nn.LayerNorm(64)
ln(x)