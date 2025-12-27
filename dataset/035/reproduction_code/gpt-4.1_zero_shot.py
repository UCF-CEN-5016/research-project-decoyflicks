import torch
import math

class RotaryPositionalEmbeddings:
    def __init__(self, dim):
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer = inv_freq

    def get_embed(self, seq_len):
        t = torch.arange(seq_len, dtype=torch.float)
        freqs = torch.einsum('i , j -> i j', t, self.register_buffer)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb

rotary_pe = RotaryPositionalEmbeddings(3)
embeddings = rotary_pe.get_embed(2)
print(embeddings)