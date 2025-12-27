import torch
from torch.nn.functional import gelu

# Create a random tensor, e.g., for input
x = torch.randn(2, 3)

# Trigger the TypeError by passing True (bool) instead of str or None to 'approximate'
try:
    gelu(x, approximate=True)
except TypeError as e:
    print(f"Caught error: {e}")

import torch
from torch.nn.functional import gelu

x = torch.randn(2, 3)
try:
    gelu(x, approximate=True)
except TypeError as e:
    print(f"Caught error: {e}")