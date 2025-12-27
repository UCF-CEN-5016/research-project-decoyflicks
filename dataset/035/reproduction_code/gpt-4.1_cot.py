# 1. Necessary imports
from labml_nn.transformers.rope import RotaryPositionalEmbeddings
import torch

# 2. Minimal setup:
# The bug is about initializing RotaryPositionalEmbeddings with a wrong feature dimension (3 instead of 4)
# and then trying to use it with input tensors of feature size 4.

# 3. Triggering conditions:
# Initialize with dim=3 but input tensor has last dimension 4

rotary_pe = RotaryPositionalEmbeddings(3)  # supposed to be 4

# Create a dummy tensor with last dimension 4 (matching example)
x = torch.randn(2, 16, 4)  # batch=2, seq_len=16, feature_dim=4

# 4. Applying rotary positional embeddings should raise an error
rotated = rotary_pe(x)