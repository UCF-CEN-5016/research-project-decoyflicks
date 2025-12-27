import torch
import x_transformers

# Set up dummy inputs
batch_size, seq_len = 2, 8
head_dim = 64
num_heads = 8
seq_len = 8

# Create dummy input tensor
x = torch.randn(batch_size, seq_len, head_dim)

# Create a 4D attention bias tensor (custom alibi positions)
# Shape: (batch_size, num_heads, seq_len, seq_len)
custom_alibi = torch.randn(batch_size, num_heads, seq_len, seq_len)

# Initialize the attention module with flash attention enabled
attn = x_transformers.Attend(
    dim=head_dim,
    num_heads=num_heads,
    causal=True,
    attn_flash=True
)

# Trigger the bug by passing custom_alibi with attn_flash=True
# The error occurs because the code expects a 3D bias but receives a 4D one
output = attn(x, bias=custom_alibi)