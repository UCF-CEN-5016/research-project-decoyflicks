import torch
import torch.nn as nn
from x_transformers import Transformer, Decoder, AutoregressiveWrapper

# Minimal setup to reproduce the bug
class SimpleTransformer(nn.Module):
    def __init__(self, dim=64, depth=2, seq_len=256):
        super().__init__()
        self.encoder = Transformer(dim=dim, depth=depth, seq_len=seq_len)
        self.decoder = Decoder(dim=dim, depth=depth, seq_len=seq_len)

    def forward(self, x):
        encodings = self.encoder(x)
        return self.decoder(encodings)

# Create a toy model and input
model = SimpleTransformer()
src = torch.randint(0, 10, (1, 256))  # Input sequence of length 256
ENC_SEQ_LEN = 256  # Length of the input sequence

# Simulate the generation call that triggers the bug
# The generated sequence length is shorter than the input
start_tokens = torch.tensor([0])  # Starting token for generation

# This call would trigger the bug as the generated sequence length is shorter than ENC_SEQ_LEN
# Note: The actual generation logic would need to be implemented in the model
# This is a simplified example to demonstrate the setup