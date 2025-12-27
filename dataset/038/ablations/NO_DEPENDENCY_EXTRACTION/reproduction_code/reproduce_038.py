import torch
import torch.nn as nn
import math

class RotaryPositionalEmbedding:
    def __init__(self, n_channels: int):
        self.n_channels = n_channels
        self.cos_cached = torch.cos(torch.arange(0, n_channels, dtype=torch.float32).view(1, -1, 1, 1))
        self.sin_cached = torch.sin(torch.arange(0, n_channels, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor, neg_half_x: torch.Tensor):
        x_rope = (x * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])
        return x_rope

batch_size = 4
n_channels = 4
x = torch.rand(batch_size, n_channels, 32, 32)
neg_half_x = torch.rand(batch_size, n_channels, 1, 4)

rotary_pos_emb = RotaryPositionalEmbedding(n_channels)
try:
    output = rotary_pos_emb.forward(x, neg_half_x)
except RuntimeError as e:
    print(e)