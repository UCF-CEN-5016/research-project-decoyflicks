import torch
from x_transformers import TransformerWrapper

# Settings
BATCH_SIZE = 1
ENC_SEQ_LEN = 258  # Encoding sequence length (longer)
DEC_SEQ_LEN = 230  # Decoding sequence length (shorter)
DIM = 32
HEADS = 4
DEPTH = 2
VOCAB_SIZE = 50

# Toy Transformer model with encoder-decoder architecture
model = TransformerWrapper(
    dim=DIM,
    depth=DEPTH,
    heads=HEADS,
    vocab_size=VOCAB_SIZE,
    max_seq_len=max(ENC_SEQ_LEN, DEC_SEQ_LEN),
    cross_attend=True,
    use_encodings=True
)

# Random source sequence (encoding)
src = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, ENC_SEQ_LEN))

# Source mask (all valid)
src_mask = torch.ones(BATCH_SIZE, ENC_SEQ_LEN, dtype=torch.bool)

# Start tokens for decoding (length shorter than encoding)
start_tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, DEC_SEQ_LEN))

# The bug: generate with decoding length shorter than encoding length triggers mask size mismatch
# Expected to raise RuntimeError about tensor size mismatch in attention mask
sample = model.generate(src, start_tokens, DEC_SEQ_LEN, mask=src_mask)

print("Generated sample shape:", sample.shape)