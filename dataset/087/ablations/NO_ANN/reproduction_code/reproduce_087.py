import torch
from x_transformers import Attention, AlibiPositionalBias

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 2
sequence_length = 10
dim = 64
heads = 8
total_heads = 8

queries = torch.randn(batch_size, sequence_length, dim, device=device)
keys = torch.randn(batch_size, sequence_length, dim, device=device)
values = queries

attention = Attention(dim=dim, heads=heads, causal=True, flash=True)
alibi_pos_bias = AlibiPositionalBias(heads=heads, total_heads=total_heads)

custom_positions = torch.randn(batch_size, sequence_length, 3, device=device)

output, _ = attention(queries, keys, values, mask=None, attn_bias=None, pos=custom_positions)

assert output.shape == (batch_size, sequence_length, dim)
assert not torch.isnan(output).any()
print(output)