import torch

x = torch.randn(2, 2)
torch.nn.functional.gelu(x, approximate=True)