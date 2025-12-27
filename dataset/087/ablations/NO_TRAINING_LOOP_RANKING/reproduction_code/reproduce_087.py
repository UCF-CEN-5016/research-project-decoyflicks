import torch
from x_transformers import AlibiPositionalBias, Attention

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 2
sequence_length = 10
dim = 64
heads = 8
causal = True

alibi_positional_bias = AlibiPositionalBias(heads=heads, total_heads=heads).to(device)
query = torch.randn(batch_size, sequence_length, dim).to(device)
key = torch.randn(batch_size, sequence_length, dim).to(device)
custom_positions = torch.randn(batch_size, sequence_length, sequence_length).to(device)

attn_flash = True
attn_bias = None

attention = Attention(dim=dim, heads=heads, causal=causal, flash=attn_flash).to(device)
output = attention.forward(query, key, key, mask=None, attn_bias=alibi_positional_bias.forward_custom_pos(custom_positions))

assert output.shape == (batch_size, sequence_length, dim)
assert not torch.isnan(output).any()

print("Test completed, bug reproduction attempted.")