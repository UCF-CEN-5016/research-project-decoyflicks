import torch

def gelu(x, approximate='tanh'):
    if approximate == 'tanh':
        return torch.nn.functional.gelu(x)
    elif approximate == 'sigmoid':
        return torch.nn.functional.gelu(x, approximate=False)
    else:
        raise ValueError(f"Unknown approximate method: {approximate}")

x = torch.randn(2, 2)
gelu(x, approximate=True)