import torch
from x_transformers import AlibiPositionalBias, flash_attn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 2
seq_length = 10
dim = 64
heads = 8

alibi_positional_bias = AlibiPositionalBias(heads=8, total_heads=8).to(device)
query = torch.randn(batch_size, seq_length, dim).to(device)
key = torch.randn(batch_size, seq_length, dim).to(device)
custom_pos = torch.randn(batch_size, seq_length, 1).to(device)

attn_flash = True
bias = alibi_positional_bias.forward_custom_pos(custom_pos, custom_pos)

assert bias.shape == (batch_size, 1, seq_length, seq_length)

attn_logits = torch.randn(batch_size, heads, seq_length, seq_length).to(device)
output, intermediates = flash_attn(query, key, key, attn_bias=bias)

assert not torch.isnan(output).any()
print('Output shape:', output.shape)
assert output.shape == (batch_size, heads, seq_length, dim)