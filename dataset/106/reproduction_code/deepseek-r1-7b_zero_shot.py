import torch

# Create tensors with values large enough to cause overflow
x = torch.tensor([1e20], dtype=torch.float64)
y = torch.tensor([1e20], dtype=torch.float64)

# Compute pairwise distances which may result in NaN due to precision issues
dist = (x.pow(2) + y.pow(2)) - 2 * x * y

print(dist)

import torch

x = torch.tensor([1e20], dtype=torch.float64)
y = torch.tensor([1e20], dtype=torch.float64)

dist = (x.pow(2) + y.pow(2)) - 2 * x * y
dist_clamped = torch.clamp(dist, min=0.0)
print(dist_clamped)