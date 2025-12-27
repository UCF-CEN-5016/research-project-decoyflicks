import torch
from x_transformers import TransformerWrapper, Decoder

# Define a toy model
model = TransformerWrapper(
    num_tokens=256,
    max_seq_len=512,
    attn_layers=Decoder(
        dim=256,
        depth=2,
        heads=8,
    ),
)

# Define input sequence
src = torch.randint(0, 256, (1, 230))  # encoding sequence

# Define start tokens for decoding
start_tokens = torch.randint(0, 256, (1, 1))

# Define mask for input sequence
src_mask = torch.ones((1, 1, 230, 230))

# Attempt to generate decoding sequence shorter than encoding sequence
try:
    sample = model.generate(src, start_tokens, 258, mask=src_mask)
except RuntimeError as e:
    print(f"Error: {e}")