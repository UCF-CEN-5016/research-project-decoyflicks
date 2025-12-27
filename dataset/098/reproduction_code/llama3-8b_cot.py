import torch
import math

# Define the SinusoidalPosEmb class
class SinusoidalPosEmb(nn.Module):
    # ... (rest of the code)

# Test script to trigger the bug
emb = SinusoidalPosEmb(dim=4)
x = torch.ones((10))
out = emb(x)
print(out, out.shape)

import torch
import math
from torch import nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(theta) / (half_dim - 1)
        # ... (rest of the code)

emb = SinusoidalPosEmb(dim=4)
x = torch.ones((10))
out = emb(x)
print(out, out.shape)