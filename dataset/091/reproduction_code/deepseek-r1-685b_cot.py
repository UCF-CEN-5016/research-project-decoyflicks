import torch
from x_transformers import XTransformer

# Setup minimal environment
ENC_SEQ_LEN = 258
DEC_SEQ_LEN = 230  # Shorter than encoder sequence
DIM = 512
VOCAB_SIZE = 1000

# Create model
model = XTransformer(
    dim=DIM,
    enc_num_tokens=VOCAB_SIZE,
    enc_depth=3,
    enc_heads=8,
    dec_num_tokens=VOCAB_SIZE,
    dec_depth=3,
    dec_heads=8,
)

# Create dummy inputs
src = torch.randint(0, VOCAB_SIZE, (1, ENC_SEQ_LEN))
src_mask = torch.ones(1, ENC_SEQ_LEN).bool()
start_tokens = torch.randint(0, VOCAB_SIZE, (1, 1))

# Trigger the bug - try to generate shorter sequence than encoder length
with torch.no_grad():
    sample = model.generate(
        src, 
        start_tokens, 
        DEC_SEQ_LEN, 
        mask=src_mask
    )