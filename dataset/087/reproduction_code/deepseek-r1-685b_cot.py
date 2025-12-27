import torch
from x_transformers import Encoder
from x_transformers.attend import FlashAttention

# Setup minimal flash attention with ALiBi
attn = FlashAttention(
    dropout=0.1,
    causal=False,
    flash=True
)

# Create dummy inputs with custom positions
batch_size = 2
seq_len = 10
dim = 64
heads = 4

q = torch.randn(batch_size, seq_len, dim)
k = torch.randn(batch_size, seq_len, dim)
v = torch.randn(batch_size, seq_len, dim)

# Custom positions that trigger 4D bias
custom_pos = torch.arange(seq_len).expand(batch_size, seq_len)

# This will fail because the 4D bias isn't properly handled in flash attention
try:
    out = attn(q, k, v, pos_bias=None, alibi_pos_bias=custom_pos)
    print("Success!")
except Exception as e:
    print(f"Failed with error: {e}")

# For comparison - non-flash attention should work
attn_no_flash = FlashAttention(
    dropout=0.1,
    causal=False,
    flash=False
)

try:
    out = attn_no_flash(q, k, v, pos_bias=None, alibi_pos_bias=custom_pos)
    print("Non-flash version worked!")
except Exception as e:
    print(f"Non-flash also failed: {e}")