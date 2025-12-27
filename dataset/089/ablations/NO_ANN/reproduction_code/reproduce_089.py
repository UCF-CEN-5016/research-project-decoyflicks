import torch
from x_transformers import Attention

torch.manual_seed(42)

dim = 64
heads = 8
kv_heads = 4
dim_head = 16
batch_size = 2

input_tensor = torch.randn(batch_size, 10, dim)
attention_layer = Attention(dim=dim, heads=heads, kv_heads=kv_heads, qk_norm=True)

output1 = attention_layer(input_tensor)
mask = torch.ones(batch_size, 10, dtype=torch.bool)
output2 = attention_layer(input_tensor, mask=mask)

assert not torch.equal(output1, output2)
print(output1.shape, output2.shape)