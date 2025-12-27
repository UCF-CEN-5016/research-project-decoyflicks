import torch
from torch import cdist

x = torch.tensor([1e20], dtype=torch.float32)
y = torch.tensor([-1e20], dtype=torch.float32)
dist = cdist(x.unsqueeze(0), y.unsqueeze(0))
print(dist)