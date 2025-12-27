import torch

# Sample data
x = torch.randn(10, 10)
y = torch.randn(10, 10)

# Reproduce the issue
dist = (x.unsqueeze(2) - y.unsqueeze(0)) ** 2
dist = dist.sum(-1)

# This will cause NaN
print("Distance matrix without clamp:", dist)

# Fix the issue with clamp
dist_clamped = torch.clamp((x.unsqueeze(2) - y.unsqueeze(0)) ** 2, min=0).sum(-1)
print("Distance matrix with clamp:", dist_clamped)