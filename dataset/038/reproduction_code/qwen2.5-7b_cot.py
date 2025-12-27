import torch

# Create x with shape (batch_size=1, seq_len=2, num_heads=3, features=4)
x = torch.randn(1, 2, 3, 4)

# Create cos_cached and sin_cached with shape (seq_len=2, num_heads=3, features=4)
cos_cached = torch.randn(2, 3, 4)
sin_cached = torch.randn(2, 3, 4)

# Slice x to remove the last feature dimension
x_rope = x[..., :3]

# Perform element-wise multiplication with cos_cached and sin_cached
result = x_rope * cos_cached[:, :, :3] + torch.randn_like(x_rope) * sin_cached[:, :, :3]