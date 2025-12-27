import einops
import torch
x = torch.randn(1, 32)
einops.rearrange(x.unsqueeze(0), 'n d -> n 1')