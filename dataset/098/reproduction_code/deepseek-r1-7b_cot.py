import math
import torch
from torch.nn import Module

class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta  # Add this line to store theta as an attribute
    
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # ... rest of the code ...

import math
import torch
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import SinusoidalPosEmb

# Correct version with theta assigned as a member variable
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # ... rest of the code ...

# Initialize and use the module
emb = SinusoidalPosEmb(dim=4, theta=10000)
x = torch.ones((10))
out = emb(x)

print(out, out.shape)