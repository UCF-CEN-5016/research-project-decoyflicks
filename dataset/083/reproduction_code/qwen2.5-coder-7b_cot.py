import torch
import torch.nn as nn

from vit_pytorch import CrossAttention

# Ensure CUDA is available; else, fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model / tensor Parameters
batch_size = 2
seq_len_q = 16
seq_len_ctx = 12
n_heads = 8
# Intentionally choose a model dimension that is fine for the query but make the context have a mismatched embedding
d_model = 256  # query / attention dim
d_model_mismatch = d_model + 1  # context has a wrong dim to trigger the bug in cross-attention

# Create dummy tensors
x_q = torch.randn(batch_size, seq_len_q, d_model).to(device)           # query tokens
context = torch.randn(batch_size, seq_len_ctx, d_model_mismatch).to(device)  # context tokens with wrong dim

# Initialize CrossAttention from vit_pytorch
cross_attn = CrossAttention(dim=d_model, heads=n_heads).to(device)

# Attempt to run cross-attention with mismatched context dimension to reproduce the bug
try:
    out = cross_attn(x_q, context)
    print("Cross-attention output shape:", out.shape)
except Exception as e:
    print(f"Bug triggered: {e}")