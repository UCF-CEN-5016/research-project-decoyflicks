import torch
from einops import rearrange

class RotaryXPos:
    def __init__(self, scale=1.0):
        self.scale = scale

    def forward(self, power):
        return self.scale ** rearrange(power, 'n -> n 1')

power = torch.randn(1, 32)
rotary_xpos = RotaryXPos()
rotary_xpos.forward(power)