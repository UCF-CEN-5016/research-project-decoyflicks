import torch

# Define batch size and feature dimensions
batch_size = 64
height = 32
width = 32
d = 4

# Create random input data x with shape (batch_size, height, width, d)
x = torch.randn(batch_size, height, width, d)

# Define the dimensions for cos_cached and sin_cached to match those in x (height=32, width=1, d=4)
cos_cached = torch.randn(height, 1, d)
sin_cached = torch.randn(height, 1, d)

# Calculate neg_half_x as -0.5 * x
neg_half_x = -0.5 * x

# Call the rotary positional embedding statement with the given equation
x_rope = (neg_half_x[:, :, :, :d // 2] * cos_cached) + (neg_half_x[:, :, :, d // 2:] * sin_cached)

# Verify that an error occurs due to tensor size mismatch between x_rope and cos_cached/sin_cached
try:
    print(x_rope)
except RuntimeError as e:
    print(e)