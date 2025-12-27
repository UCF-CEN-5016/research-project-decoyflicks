import torch

# Define tensors with incompatible shapes
x = torch.randn(3, 4, 2, 2)  # shape: (batch, seq_len, num_heads, head_dim)
cos_cached = torch.randn(4, 2, 2)  # shape: (seq_len, num_heads, head_dim)
sin_cached = torch.randn(4, 2, 2)

# Simulate the incorrect operation
x_rope = x  # Assume this is a part of x
neg_half_x = x  # Placeholder for neg_half_x

# Incorrect slicing: slices along the batch dimension instead of the sequence length
cos_cached_slice = cos_cached[:x.shape[0]]  # shape: (3, 2, 2)
sin_cached_slice = sin_cached[:x.shape[0]]  # shape: (3, 2, 2)

# This line would raise an error due to shape mismatch
result = x_rope * cos_cached_slice + neg_half_x * sin_cached_slice

# Correct slicing: slices along the sequence length dimension
cos_cached_slice = cos_cached[:x.shape[1]]  # shape: (4, 2, 2)
sin_cached_slice = sin_cached[:x.shape[1]]  # shape: (4, 2, 2)

# Now the shapes match, and the operation is valid
result = x_rope * cos_cached_slice + neg_half_x * sin_cached_slice