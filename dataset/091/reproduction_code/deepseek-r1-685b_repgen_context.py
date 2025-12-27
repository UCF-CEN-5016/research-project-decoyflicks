import torch
from x_transformers import TransformerWrapper, Encoder, Decoder

# Model setup
model = TransformerWrapper(
    num_tokens=256,
    max_seq_len=512,
    attn_layers=Decoder(
        dim=512,
        depth=6,
        heads=8,
        cross_attend=True
    )
).cuda()

# Sequence lengths
ENC_SEQ_LEN = 230  # Encoding sequence length
DEC_SEQ_LEN = 200  # Decoding sequence length (shorter than encoding)

# Sample data
src = torch.randint(0, 256, (1, ENC_SEQ_LEN)).cuda()
start_tokens = torch.randint(0, 256, (1, 1)).cuda()
src_mask = torch.ones(1, ENC_SEQ_LEN).bool().cuda()

# Generate sequence
try:
    sample = model.generate(src, start_tokens, DEC_SEQ_LEN, mask=src_mask)
except RuntimeError as e:
    print(f"An error occurred: {e}")