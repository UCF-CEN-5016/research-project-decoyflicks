import torch
from x_transformers import XTransformer

# Step 1: Create a random input tensor
batch_size = 1
seq_length = 5
input_tensor = torch.randn(batch_size, seq_length)

# Step 2: Define the pad_id that shouldn't be used in padding (e.g., non-zero)
pad_id = 42

# Step 3: Wrap the model and call align_right with the specified pad_id
wrapped_model = XTransformer()
result = wrapped_model.align_right(input_tensor, pad_id=pad_id)

# Verification would check if F.pad uses 'value' as pad_id; in current setup, it's still 0.