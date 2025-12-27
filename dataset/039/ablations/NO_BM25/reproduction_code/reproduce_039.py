import torch
import numpy as np
from labml_nn.transformers.rope.value_pe import RotaryValuePEMultiHeadAttention

# Setup
batch_size = 4
seq_len = 3
d_model = 8
heads = 2

# Create input tensors
query = torch.rand(seq_len, batch_size, d_model)
key = torch.rand(seq_len, batch_size, d_model)
value = torch.rand(seq_len, batch_size, d_model)
mask = torch.ones(seq_len, seq_len, batch_size)

# Instantiate the model
model = RotaryValuePEMultiHeadAttention(heads=heads, d_model=d_model, rope_percentage=0.5, dropout_prob=0.0)

# Call forward method
output = model(query=query, key=key, value=value, mask=mask)

# Modify the line in ReverseRotaryPositionalEmbeddings class
# x_rope = (x_rope * self.cos_cached[:x.shape[0]]) - (neg_half_x * self.sin_cached[:x.shape[0]])

# Re-run the forward method to trigger the bug
try:
    output = model(query=query, key=key, value=value, mask=mask)
except RuntimeError as e:
    print(e)