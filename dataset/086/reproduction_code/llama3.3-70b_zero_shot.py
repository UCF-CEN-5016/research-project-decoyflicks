import torch
from einops import rearrange

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, power):
        scale = self.scale ** rearrange(power, 'n -> n 1')
        return scale

rotary_embedding = RotaryEmbedding(dim=32)
power = torch.randn(1, 32)
rotary_embedding(power)