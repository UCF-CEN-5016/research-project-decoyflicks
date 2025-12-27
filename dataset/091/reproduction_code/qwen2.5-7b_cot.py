import torch
import torch.nn as nn
from x_transformers import Transformer, Decoder, AutoregressiveWrapper

class SimpleTransformer(nn.Module):
    def __init__(self, dim=64, depth=2, seq_len=256):
        super().__init__()
        self.encoder = Transformer(dim=dim, depth=depth, seq_len=seq_len)
        self.decoder = Decoder(dim=dim, depth=depth, seq_len=seq_len)

    def forward(self, x):
        encodings = self.encoder(x)
        return self.decoder(encodings)

def simulate_generation_bug(model, src, start_tokens, gen_seq_len):
    with torch.no_grad():
        encodings = model.encoder(src)
        # Simulate generating a sequence with length gen_seq_len
        generated_seq = torch.randint(0, 10, (1, gen_seq_len))
        return model.decoder(encodings, generated_seq)

# Create a toy model and input
model = SimpleTransformer()
src = torch.randint(0, 10, (1, 256))  # Input sequence of length 256

# Constants
ENC_SEQ_LEN = 256  # Length of the input sequence

# Simulate the generation call that triggers the bug
start_tokens = torch.tensor([0])  # Starting token for generation
GEN_SEQ_LEN = 128  # Length of the generated sequence

result = simulate_generation_bug(model, src, start_tokens, GEN_SEQ_LEN)