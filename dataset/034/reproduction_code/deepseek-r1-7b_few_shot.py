import torch
from torch.nn.functional import gelu

# Example usage causing TypeError:
x = torch.randn(5)
try:
    gelu(x, approximate=True)  # This will throw the error if 'approximate' expects a string
except TypeError as e:
    print(f"Error: {e}")

print(gelu(x, approximate="tanh"))  # This works as intended