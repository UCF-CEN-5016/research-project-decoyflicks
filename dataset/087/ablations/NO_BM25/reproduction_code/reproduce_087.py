import torch
import einx
from x_transformers import Attention

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 2
sequence_length = 10
dim = 64
heads = 8
max_pos = 16

query_tensor = torch.rand(batch_size, sequence_length, dim, device=device)
context_tensor = torch.rand(batch_size, sequence_length, dim, device=device)
pos_i = torch.rand(batch_size, sequence_length, 1, device=device)
pos_j = torch.rand(batch_size, sequence_length, 1, device=device)

attention_layer = Attention(dim=dim, heads=heads, flash=True, causal=True)
output = attention_layer.forward(query_tensor, context=context_tensor, attn_mask=None, rel_pos=None, attn_bias=None, pos=pos_i)

assert output.shape == (batch_size, sequence_length, dim)
assert not torch.isnan(output).any()

bias_tensor = attention_layer.bias
assert bias_tensor.shape == (batch_size, heads, sequence_length, sequence_length)