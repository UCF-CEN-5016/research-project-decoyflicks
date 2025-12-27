import torch
from x_transformers import Encoder, Decoder, DecoderWrapper, EncoderDecoder

# Settings
BATCH_SIZE = 1
ENC_SEQ_LEN = 258  # Encoding sequence length (longer)
DEC_SEQ_LEN = 230  # Decoding sequence length (shorter)
DIM = 16
HEADS = 4
LAYS = 2
VOCAB_SIZE = 50

# Create toy encoder and decoder
encoder = Encoder(dim=DIM, depth=LAYS, heads=HEADS, vocab_size=VOCAB_SIZE)
decoder = Decoder(dim=DIM, depth=LAYS, heads=HEADS, causal=True, vocab_size=VOCAB_SIZE)
decoder = DecoderWrapper(decoder)  # wraps decoder for autoregressive decoding

model = EncoderDecoder(encoder, decoder)

# Random source input (encoding)
src = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, ENC_SEQ_LEN))

# Source mask (mask out nothing, all True)
src_mask = torch.ones(BATCH_SIZE, ENC_SEQ_LEN).bool()

# Encoding pass
with torch.no_grad():
    encodings = model.encoder(src, mask=src_mask)

# Start tokens for decoding (batch size and start tokens length)
start_tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 1))

# Generate shorter sequence than encoding length
with torch.no_grad():
    # Here seq_len < ENC_SEQ_LEN
    output = model.generate(src, start_tokens, seq_len=DEC_SEQ_LEN, mask=src_mask)

print("Output shape:", output.shape)
print("Output tokens:", output)