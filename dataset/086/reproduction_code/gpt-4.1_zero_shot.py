import torch
from einops import rearrange

power = torch.randn(1, 32)
scale = 2.0 ** rearrange(power, 'n -> n 1')