import torch
import torch.nn.functional as F

def gelu(x):
    return F.gelu(x, approximate=True)

x = torch.randn(3)
gelu(x)