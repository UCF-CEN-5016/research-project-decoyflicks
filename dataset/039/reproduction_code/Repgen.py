import torch

# Create tensors with mismatched dimensions
x = torch.randn(2, 4, 3)  # Last dimension is 3
cos_cached = torch.randn(4, 1, 4)  # Last dimension is 4
sin_cached = torch.randn(4, 1, 4)  # Last dimension is 4

# Create rotary embedding inputs
x_rope = x
neg_half_x = -0.5 * x

# This line will fail with dimension mismatch error
result = (x_rope * cos_cached[:x.shape[1]]) + (neg_half_x * sin_cached[:x.shape[1]])