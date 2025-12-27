import torch

# Define parameters
x = torch.zeros(3, 128)
cos_cached = torch.zeros(3, 128, 4096, 4096)
sin_cached = torch.zeros(3, 128, 4096, 4096)

# Compute neg_half_x
neg_half_x = x * -0.5

# Slice cos_cached and sin_cached
cos_cached_slice = cos_cached[:, :, :, :3]
sin_cached_slice = sin_cached[:, :, :, :3]

# Define the variable 'x_rope' as it is used in the original code but not defined
x_rope = x

# Compute final result
result = (x_rope * cos_cached_slice) + (neg_half_x * sin_cached_slice)