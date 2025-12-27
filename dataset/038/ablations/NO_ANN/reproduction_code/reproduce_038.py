import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos_cached = torch.rand(4, 1, 1, 4)
        self.sin_cached = torch.rand(4, 1, 1, 4)

    def forward(self, x):
        neg_half_x = torch.rand(4, 1, 1, 4)
        x_rope = (x * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])
        return x_rope

batch_size = 4
channels = 4
x = torch.rand(batch_size, channels, 32, 32)

rotary_pos_emb = RotaryPositionalEmbedding()
output = rotary_pos_emb(x)