import torch
import torch.nn as nn
import math

class RotaryValueEmbeddings(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.theta = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_emb(self, x, seq_len):
        position = torch.arange(seq_len, device=x.device).float()
        freqs = torch.einsum('i,j->ij', position, self.theta.to(x.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return x * emb.cos() + self.rotate_half(x) * emb.sin()

    def forward(self, x):
        seq_len = x.shape[1]
        x = self.apply_rotary_emb(x, seq_len)  # First rotation
        x = self.apply_rotary_emb(x, seq_len)  # Second rotation
        return x

d_model = 64
seq_len = 10
batch_size = 2
rope = RotaryValueEmbeddings(d_model)
x = torch.randn(batch_size, seq_len, d_model)
output = rope(x)