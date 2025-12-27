import torch

# Create a tensor x with shape (batch, seq_len, num_heads, head_dim)
x = torch.randn(3, 4, 2, 5)  # batch=3, seq_len=4, num_heads=2, head_dim=5
x_rope = x.clone()  # Create a copy of x

# Create cos_cached with shape (max_seq_len, num_heads, head_dim)
cos_cached = torch.randn(5, 2, 5)  # max_seq_len=5

# Correct slicing along the first dimension (max_seq_len)
cos_slice = cos_cached[:x.shape[1]]  # shape (4, 2, 5)

# Now, multiply x_rope with cos_slice
result = x_rope * cos_slice