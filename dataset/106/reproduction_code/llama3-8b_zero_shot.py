import torch
import numpy as np

x = torch.tensor([1e20], dtype=torch.float32)
y = torch.tensor([1e20-1e-5], dtype=torch.float32)

dist = (x-y)**2
print(dist)