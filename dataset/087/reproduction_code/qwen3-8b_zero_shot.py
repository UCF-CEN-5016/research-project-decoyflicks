import torch

# Step 1: Generate a 4D tensor (alibi_pos)
seq_len = 5
alibi_pos = torch.arange(seq_len).view(1, 1, seq_len, 1).repeat(1, 8, 1, seq_len)

# Step 2: Generate a 3D tensor (dummy_tensor)
dummy_tensor = torch.rand(8, seq_len, seq_len)

# Step 3: Attempt to perform element-wise addition
# This will raise a shape mismatch error
result = dummy_tensor + alibi_pos