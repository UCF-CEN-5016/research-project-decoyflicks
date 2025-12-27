import torch

# Define tensors with incompatible shapes
x = torch.randn(3, 4, 2, 2)  # shape: (batch, seq_len, num_heads, head_dim)
cos_cached = torch.randn(4, 2, 2)  # shape: (seq_len, num_heads, head_dim)
sin_cached = torch.randn(4, 2, 2)

# Simulate the incorrect operation (bug: slicing with x.shape[0] instead of seq dim)
x_rope = x[:3]  # Assume this is a part of x
neg_half_x = x  # Placeholder for neg_half_x

# Incorrect slicing: uses batch size instead of sequence length
cos_cached_slice = cos_cached[:x.shape[0]]  # INCORRECT: shape (3, 2, 2)
sin_cached_slice = sin_cached[:x.shape[0]]  # INCORRECT: shape (3, 2, 2)

# This will raise a RuntimeError due to mismatched dimensions during broadcasting
result = x_rope * cos_cached_slice + neg_half_x * sin_cached_slice