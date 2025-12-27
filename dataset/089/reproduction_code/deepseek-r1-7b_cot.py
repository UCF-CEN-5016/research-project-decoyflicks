self.qk_norm_k_scale = nn.Parameter(torch.ones(kv_heads, 1, dim_head))

import torch
from x_transformers import XTransformer, QKVAttention

dim_head = 64
qk_norm = 3  # Number of query/key normalization factors
kv_heads = 2  # Number of key-value attention heads
model = XTransformer(
    dim=1024,
    seq_len=512,
    qk_norm=qk_norm,
    kv_heads(kv_heads),
    attend_mask=True,
)

batch_size = 8
seq_len = 512

# Random input tensor
x = torch.randn(batch_size, seq_len, 1024)

try:
    with torch.no_grad():
        output = model(x)
except Exception as e:
    print(f"Error: {e}")