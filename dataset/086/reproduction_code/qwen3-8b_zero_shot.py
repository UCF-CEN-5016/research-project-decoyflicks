import torch
import einops

x = torch.randn(1, 32)
einops.rearrange(x, 'n -> n 1')