import torch
import torch.nn.functional as F

x = torch.tensor([1.0])
F.gelu(x, approximate=True)