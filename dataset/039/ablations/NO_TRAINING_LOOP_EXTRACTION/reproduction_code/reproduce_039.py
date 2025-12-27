import torch
from labml_nn.transformers.rope import ReverseRotaryPositionalEmbeddings

seq_len = 4
batch_size = 2
n_heads = 2
d_model = 4

x = torch.randn(seq_len, batch_size, n_heads, d_model)
reverse_rope = ReverseRotaryPositionalEmbeddings(d_model)

try:
    output = reverse_rope(x)
except RuntimeError as e:
    assert str(e) == "The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 3"