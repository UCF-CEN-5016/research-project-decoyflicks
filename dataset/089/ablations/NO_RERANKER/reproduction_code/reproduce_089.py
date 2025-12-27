import torch
import torch.nn as nn
from x_transformers import Attention

torch.manual_seed(42)

dim = 64
heads = 8
kv_heads = 4
dim_head = 16
qk_norm = True

batch_size = 2
seq_length = 10
input_tensor = torch.randn(batch_size, seq_length, dim)

attention_layer = Attention(dim=dim, heads=heads, kv_heads=kv_heads, dim_head=dim_head, qk_norm=qk_norm)
output = attention_layer(input_tensor, mask=None)

assert output.shape == (batch_size, seq_length, dim), "Output shape mismatch"

qk_norm_k_scale = attention_layer.qk_norm_k_scale
assert qk_norm_k_scale.shape == (heads, 1, dim_head), "qk_norm_k_scale shape mismatch"

print(qk_norm_k_scale.shape)