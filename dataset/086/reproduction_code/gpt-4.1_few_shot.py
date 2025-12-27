import torch
from einops import rearrange

# Simulate a power tensor that should be 1D but is 2D due to prior hack
power = torch.randn(1, 32)  # Shape: [1, 32]

# Attempt to rearrange with pattern expecting 1D input -> Error
# This mimics the rotary_xpos bug where a 2D tensor is passed to 'n -> n 1'
scale = 1000 ** rearrange(power, 'n -> n 1')  # Should raise einops.EinopsError