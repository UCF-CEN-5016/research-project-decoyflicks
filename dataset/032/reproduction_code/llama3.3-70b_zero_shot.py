import torch

def gelu(x):
    return torch.nn.functional.gelu(x, approximate=True)

x = torch.randn(1, 10)
gelu(x)