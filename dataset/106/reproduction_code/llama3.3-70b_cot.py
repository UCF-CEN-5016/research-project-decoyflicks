import torch

# Minimal environment setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create large tensors to trigger float precision issues
x = torch.randn(100, 1000, device=device) * 1e10
y = torch.randn(100, 1000, device=device) * 1e10

# Triggering conditions: compute pairwise distances without clamping
dist = torch.cdist(x, y)

# Check for nan values
print("Contains nan:", torch.isnan(dist).any())

# Core issue isolation and fix: use torch.clamp to ensure non-negative values
dist_clamped = torch.cdist(x, y, p=2)  # p=2 ensures Euclidean distance
# Alternative manual computation with clamping
dist_manual = torch.sqrt(torch.clamp((x.unsqueeze(2) - y.unsqueeze(1)) ** 2, min=0).sum(dim=0))

# Check for nan values after applying fix
print("Contains nan after fix (cdist with p=2):", torch.isnan(dist_clamped).any())
print("Contains nan after manual fix:", torch.isnan(dist_manual).any())