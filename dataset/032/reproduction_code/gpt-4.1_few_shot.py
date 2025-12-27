import torch
import torch.nn.functional as F

def gelu(x):
    # Incorrect usage: approximate expects a string ('none' or 'tanh'), not a bool
    return F.gelu(x, approximate=True)

x = torch.randn(5)
try:
    y = gelu(x)
except TypeError as e:
    print(f"Caught error: {e}")