import torch
import torch.nn.functional as F

def gelu(x):
    return F.gelu(x, approximate=True)

x = torch.randn(1, 10)
gelu(x)