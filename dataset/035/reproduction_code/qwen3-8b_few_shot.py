import torch
from labml_nn.transformers.rope import RotaryPositionalEmbeddings

# Incorrect initialization with 3 features (should be 4)
rotary_pe = RotaryPositionalEmbeddings(3)

# This will raise an error due to mismatched feature dimension
# The correct initialization should be RotaryPositionalEmbeddings(4)