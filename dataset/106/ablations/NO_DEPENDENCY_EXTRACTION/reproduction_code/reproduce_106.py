import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from torch import einsum
from torch import sum as reduce  # Importing reduce as sum for clarity

def identity(t):
    return t

def l2norm(t):
    return F.normalize(t, p=2, dim=-1)

def cdist(x, y):
    # Calculate squared norms
    x2 = reduce(x ** 2, dim=1)  # Sum over the last dimension
    y2 = reduce(y ** 2, dim=1)  # Sum over the last dimension
    # Calculate pairwise squared differences
    xy = einsum('b i d, b j d -> b i j', x, y) * -2
    # Compute the distance while ensuring non-negative values
    distances = (rearrange(x2, 'b i -> b i 1') + rearrange(y2, 'b j -> b 1 j') + xy).clamp(min=0).sqrt()
    return distances

batch_size = 8
feature_dimension = 1024
# Create tensors with large values to reproduce the bug
x = torch.randn(batch_size, feature_dimension) * 1e10
y = torch.randn(batch_size, feature_dimension) * 1e10

# Compute pairwise distances
distances = cdist(x, y)
# Check for NaN values in the distances
nan_check = torch.isnan(distances).any()
print(nan_check)
if nan_check:
    print(distances)