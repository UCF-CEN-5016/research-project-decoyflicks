import torch
from x_transformers import TransformerWrapper, Decoder
from x_transformers.utils import default

# Set up minimal environment
ENC_SEQ_LEN = 230
DEC_SEQ_LEN = 258  # Decoding sequence length is longer than encoding sequence length
BATCH_SIZE = 1
VOCAB_SIZE = 1000

# Initialize a Transformer model
model = TransformerWrapper(
    num_tokens=VOCAB_SIZE,
    max_seq_len=ENC_SEQ_LEN + DEC_SEQ_LEN,
    dim=512,
    depth=6,
    heads=8,
    dim_head=64,
    ff_mult=4,
    ff_dropout=0.1,
    attn_dropout=0.1,
    decoder=Decoder(
        dim=512,
        depth=6,
        heads=8,
        dim_head=64,
        ff_mult=4,
        ff_dropout=0.1,
        attn_dropout=0.1,
    ),
)

# Create input tensors
src = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, ENC_SEQ_LEN))
src_mask = torch.ones((BATCH_SIZE, ENC_SEQ_LEN), dtype=torch.bool)
start_tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 1))

# Triggering conditions
try:
    sample = model.generate(
        src,
        start_tokens,
        DEC_SEQ_LEN,
        mask=src_mask,
    )
except RuntimeError as e:
    print(f"Error: {e}")