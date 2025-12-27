import torch

# Create x with shape (batch_size=1, seq_len=2, num_heads=3, features=4)
x = torch.randn(1, 2, 3, 4)

# Create cos_cached and sin_cached with shape (seq_len=2, num_heads=3, features=4)
cos_cached = torch.randn(2, 3, 4)
sin_cached = torch.randn(2, 3, 4)

# Create x_rope by slicing the last dimension to 3 (instead of 4)
x_rope = x[:, :, :, :3]

# Attempt to perform element-wise multiplication with cos_cached[:x.shape[0]]
# This will trigger the dimension mismatch error
try:
    result = (x_rope * cos_cached[:x.shape[0]]) + (torch.randn(x.shape[0], x.shape[1], x.shape[2], 3) * sin_cached[:x.shape[0]])
except RuntimeError as e:
    print("Error:", e)

cos_cached = cos_cached[:, :, :3]  # Shape becomes (2, 3, 3)
sin_cached = sin_cached[:, :, :3]  # Shape becomes (2, 3, 3)