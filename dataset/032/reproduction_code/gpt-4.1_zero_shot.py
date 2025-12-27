import torch

def gelu(x):
    return torch.nn.functional.gelu(x, approximate=True)

x = torch.randn(3)
gelu(x)