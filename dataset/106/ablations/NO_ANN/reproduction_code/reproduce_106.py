import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from torch import einsum

# Define a simple reduce function to sum over specified dimensions
def reduce(tensor, pattern, reduction):
    # This function will perform the specified reduction operation
    return einsum(pattern, tensor)

def cdist(x, y):
    # Calculate squared norms
    x2 = reduce(x ** 2, 'b n d -> b n', 'sum')
    y2 = reduce(y ** 2, 'b n d -> b n', 'sum')
    
    # Calculate pairwise distances
    xy = einsum('b i d, b j d -> b i j', x, y) * -2
    
    # Combine results and clamp to prevent negative values before taking the square root
    distances = (rearrange(x2, 'b i -> b i 1') + rearrange(y2, 'b j -> b 1 j') + xy)
    distances = torch.clamp(distances, min=0)  # Prevent negative values
    return distances.sqrt()

batch_size = 10
feature_dimension = 1024
# Create tensors with large values to simulate float precision overflow
x = torch.randn(batch_size, feature_dimension) * 1e10
y = torch.randn(batch_size, feature_dimension) * 1e10

# Calculate distances
distances = cdist(x, y)
# Assert that NaN values are present to reproduce the bug
assert torch.isnan(distances).any()
print("NaN values encountered in distance calculation.")