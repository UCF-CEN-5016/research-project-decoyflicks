import torch
import torch.nn as nn

class RotaryPositionalEmbedding:
    def __init__(self):
        self.cos_cached = torch.rand(4, 32, 32)
        self.sin_cached = torch.rand(4, 32, 32)

    def apply_positional_embedding(self, x):
        neg_half_x = torch.rand(4, 32, 32)
        x_rope = (x * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])
        return x_rope

batch_size = 4
input_channels = 4
height, width = 32, 32
x = torch.rand(batch_size, input_channels, height, width)

rotary_positional_embedding = RotaryPositionalEmbedding()

try:
    rotary_positional_embedding.apply_positional_embedding(x)
except RuntimeError as e:
    print(e)
    assert "The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 3" in str(e)