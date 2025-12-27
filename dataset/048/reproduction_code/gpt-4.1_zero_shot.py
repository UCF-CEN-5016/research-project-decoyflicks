import torch

a = torch.randn(10, 1649)
b = torch.randn(10, 1799)
torch.cat([a, b], dim=1).squeeze()