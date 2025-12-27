import numpy as np
import torch
from torch import nn

# Define parameters
batch_size = 32
num_heads = 8
height, width = 32, 32

# Create random input tensor and attention weights
input_tensor = torch.randn(batch_size, 3, height, width)
attn_weights = torch.randn(batch_size, num_heads, height, width)

# Simulate the attention mechanism
attn = torch.einsum('bihw,bjhw->bijh', input_tensor, attn_weights)

# Apply softmax along the j axis
attn = attn.softmax(dim=1)

# Assert that the sum of the softmax output along the j axis equals 1 for each head
assert torch.allclose(attn.sum(dim=1), torch.ones(batch_size, height, width))

# Log the output of the model
print(attn)

# Modify softmax to apply along the j axis
attn_modified = attn.softmax(dim=2)

# Assert that the outputs from both implementations are not equal
assert not torch.equal(attn, attn_modified)