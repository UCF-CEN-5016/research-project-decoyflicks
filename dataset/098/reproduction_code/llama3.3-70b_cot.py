import torch
import torch.nn as nn
import math

# Define the SinusoidalPosEmb class with the bug
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# Create an instance of the SinusoidalPosEmb class
emb = SinusoidalPosEmb(dim = 4)

# Create a tensor input
x = torch.ones((10))

# Call the forward method to trigger the bug
try:
    out = emb(x)
    print(out, out.shape)
except Exception as e:
    print(f"Error: {e}")