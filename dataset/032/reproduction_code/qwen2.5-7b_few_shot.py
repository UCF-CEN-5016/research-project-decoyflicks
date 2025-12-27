import torch
import torch.nn.functional as F

def test_gelu():
    x = torch.randn(5)
    result = F.gelu(x)  # Using the default parameters of gelu function

test_gelu()