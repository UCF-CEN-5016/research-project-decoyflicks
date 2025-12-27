import torch
from einops import rearrange

# Define a tensor with incorrect shape
power = torch.randn(1, 32)

# Attempt to rearrange the tensor
try:
    scale = rearrange(power, 'n -> n 1')
except Exception as e:
    print(f"Error: {e}")

# Expected shape should be 1-dimensional
# To fix, ensure the input tensor has the correct shape
# For example:
power_correct = torch.randn(32)
scale_correct = rearrange(power_correct, 'n -> n 1')
print(f"Correct scale shape: {scale_correct.shape}")