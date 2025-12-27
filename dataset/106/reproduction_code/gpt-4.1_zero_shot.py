import torch

x = torch.tensor([[1e10, 1e10]])
y = torch.tensor([[1e10, -1e10]])

dists = (x ** 2).sum(dim=1, keepdim=True) + (y ** 2).sum(dim=1) - 2 * x @ y.t()
print(dists)