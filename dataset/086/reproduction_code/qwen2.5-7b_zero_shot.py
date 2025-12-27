import torch
import einops

x = torch.randn(1, 32)
x = x.unsqueeze(1)