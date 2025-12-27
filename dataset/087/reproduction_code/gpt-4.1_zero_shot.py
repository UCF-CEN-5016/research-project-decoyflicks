import torch
from x_transformers import XTransformer

model = XTransformer(
    dim=16,
    depth=1,
    heads=2,
    attn_flash=True,
    max_seq_len=4,
)

custom_pos = torch.arange(4).unsqueeze(0)
x = torch.randn(1, 4, 16)
model(x, context=None, pos=custom_pos)