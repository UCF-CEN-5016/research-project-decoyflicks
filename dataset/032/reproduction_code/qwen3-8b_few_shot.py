import torch
import torch.nn.functional as F

# Incorrect usage of gelu with boolean 'approximate' argument
def test_gelu():
    x = torch.randn(5)
    result = F.gelu(x, approximate=True)  # TypeError: argument 'approximate' must be str, not bool

test_gelu()