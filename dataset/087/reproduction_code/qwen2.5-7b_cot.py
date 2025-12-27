import torch
import x_transformers

def create_inputs(batch_size, seq_len, head_dim):
    x = torch.randn(batch_size, seq_len, head_dim)
    return x

def create_custom_alibi(batch_size, num_heads, seq_len):
    custom_alibi = torch.randn(batch_size, num_heads, seq_len, seq_len)
    return custom_alibi

def initialize_attention_module(head_dim, num_heads):
    attn = x_transformers.Attend(
        dim=head_dim,
        num_heads=num_heads,
        causal=True,
        attn_flash=True
    )
    return attn

def trigger_bug(attn_module, inputs, custom_bias):
    output = attn_module(inputs, bias=custom_bias)
    return output

# Set up dummy inputs
batch_size, seq_len = 2, 8
head_dim = 64
num_heads = 8

# Create dummy input tensor
x = create_inputs(batch_size, seq_len, head_dim)

# Create a 4D attention bias tensor (custom alibi positions)
# Shape: (batch_size, num_heads, seq_len, seq_len)
custom_alibi = create_custom_alibi(batch_size, num_heads, seq_len)

# Initialize the attention module with flash attention enabled
attn = initialize_attention_module(head_dim, num_heads)

# Trigger the bug by passing custom_alibi with attn_flash=True
output = trigger_bug(attn, x, custom_alibi)