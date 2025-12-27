import torch
from torch import nn
from x_transformers import TransformerWrapper, Decoder

# Problematic configuration
model = TransformerWrapper(
    num_tokens=256,
    max_seq_len=512,
    attn_layers=Decoder(
        dim=512,
        depth=6,
        heads=8,
        kv_heads=4,  # Different from heads
        qk_norm=True,  # Enabled with kv_heads != heads
        dim_head=64
    )
)

# Forward pass with sample input
x = torch.randint(0, 256, (1, 128))
out = model(x)  # Will raise shape mismatch error

print(out.shape)