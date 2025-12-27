import torch

batch_size = 32
seq_len = 8

# Create x_rope with shape (batch, features, seq_len)
x_rope = torch.randn(batch_size, 16, seq_len)  # Shape: [32, 16, 8]

# Assume self_cos has dimensions [batch, type, seq_len, embed_dim]
self_cos = torch.randn(4, 5, 64)

# Incorrect slicing only along the first dimension (batch), resulting in shape mismatch
# This would cause an error during broadcasting when multiplying with x_rope
cos_slice_incorrect = self_cos[:x_rope.shape[0]]  # Shape: [32,5,64]

# Attempt to add tensors which have incompatible shapes due to incorrect slicing
try:
    result = cos_slice_incorrect + torch.randn(64,)  # Creates shape mismatch (4 vs. 64)
except Exception as e:
    print(f"Error during addition: {e}")