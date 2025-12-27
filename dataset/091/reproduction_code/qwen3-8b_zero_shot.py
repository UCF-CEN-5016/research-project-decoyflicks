import torch
from x_transformers import Transformer, Perceiver, Encoder, Decoder

# Minimal setup to reproduce the bug
model = Transformer(
    dim=64,
    depth=2,
    heads=4,
    encoder_depth=2,
    encoder_heads=4,
    decoder_depth=2,
    decoder_heads=4,
    num_tokens=1000,
    max_seq_len=256
)

src = torch.randint(0, 1000, (1, 256))
src_mask = torch.ones(1, 1, 256, 256)

# Attempt to generate a shorter sequence than the encoding sequence
sample = model.generate(
    src, 
    start_tokens=torch.tensor([1]), 
    seq_len=200, 
    mask=src_mask
)