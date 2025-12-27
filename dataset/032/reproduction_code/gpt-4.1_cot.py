import torch
import torch.nn.functional as F

# Create a sample tensor
x = torch.randn(3)

# This call will raise the TypeError because approximate expects a str, not bool
try:
    y = F.gelu(x, approximate=True)
except TypeError as e:
    print(f"Caught an error: {e}")