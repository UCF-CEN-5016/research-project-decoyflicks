import torch

def gelu(x, approximate=True):
    if approximate:
        return torch.nn.functional.gelu(x)
    else:
        return torch.nn.functional.gelu(x, approximate=False)

x = torch.randn(2, 2)
gelu(x, approximate=True)