import torch
from labml_nn.transformers.rope import RotaryPositionalEmbeddings

# Initialize with incorrect dimension (3 instead of 4)
rotary_pe = RotaryPositionalEmbeddings(3)  # Should be 4

# Dummy input tensor with last dimension size 4
x = torch.randn(2, 8, 4)  # (batch, seq_len, dim)

# This will raise an error due to dimension mismatch
rotated_x = rotary_pe(x)
print(rotated_x)