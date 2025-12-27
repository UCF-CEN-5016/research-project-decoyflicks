import torch
from transformers import xformers
from einops import rearrange

# Minimal setup
x = torch.randn(1, 32)

# Applying conditions that trigger the bug
rotary_xpos = xformers.RotaryXPos()
result = rearrange(rotary_xpos(x), 'n -> n 1')

print("Input shape:", x.shape)
print("Output shape after rearrangement:", result.shape)