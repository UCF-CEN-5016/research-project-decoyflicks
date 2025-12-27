import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from torch import einsum

# Define a simple reduce function to sum over specified dimensions
def reduce(tensor, pattern, reduction):
    return einsum(pattern, tensor)

def cdist(x, y):
    # Compute squared norms of x and y
    x2 = reduce(x ** 2, 'b n d -> b n', 'sum')
    y2 = reduce(y ** 2, 'b n d -> b n', 'sum')
    
    # Compute the pairwise squared distances
    xy = einsum('b i d, b j d -> b i j', x, y) * -2
    
    # Calculate the distance matrix and clamp to avoid NaN values
    distances = (rearrange(x2, 'b i -> b i 1') + rearrange(y2, 'b j -> b 1 j') + xy).sqrt()
    
    # Ensure all distances are non-negative
    distances = torch.clamp(distances, min=0)
    
    return distances

batch_size = 4
feature_dim = 1024
# Create input tensors that may lead to float precision overflow
x = torch.randn(batch_size, feature_dim) * 1e10
y = torch.randn(batch_size, feature_dim) * 1e10

# Compute distances
distances = cdist(x, y)

# Check for NaN values in the resulting distance matrix
has_nan = torch.isnan(distances).any()
print(has_nan)

# Assert that NaN values are present to reproduce the bug
assert has_nan