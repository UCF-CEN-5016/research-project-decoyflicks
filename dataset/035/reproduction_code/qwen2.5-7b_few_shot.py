import torch
from labml_nn.transformers.rope import RotaryPositionalEmbeddings

# Initialize with 4 features
rotary_pe = RotaryPositionalEmbeddings(4)

# Use the correct number of features to avoid mismatch error