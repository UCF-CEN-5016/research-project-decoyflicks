import torch
import torch.nn as nn
from x_transformers import Attention

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dim = 512
heads = 8
depth = 6
causal = True
flash = True
dropout = 0.1

x = torch.randn(2, 10, dim).to(device)
context = torch.randn(2, 10, dim).to(device)
mask = torch.ones(2, 10, dtype=torch.bool).to(device)

attention_layer = Attention(
    dim=dim,
    heads=heads,
    causal=causal,
    flash=flash,
    dropout=dropout
).to(device)

pos = torch.arange(10).repeat(2, 1, 1).to(device)

output, intermediates = attention_layer.forward(x, context=context, mask=mask, pos=pos)

assert output.shape == (2, 10, dim)
assert not torch.isnan(output).any()