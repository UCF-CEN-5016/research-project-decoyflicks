import torch

def gelu(x, approximate='tanh'):
    if approximate == 'tanh':
        return torch.nn.functional.gelu(x)
    elif approximate == 'relu':
        return torch.nn.functional.relu(x)
    else:
        raise ValueError(f"Unknown approximate method: {approximate}")

x = torch.randn(2, 3)
gelu(x, approximate='tanh')