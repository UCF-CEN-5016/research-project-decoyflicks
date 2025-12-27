import torch

# Define a function to reproduce the error
def gelu(x):
    return torch.nn.functional.gelu(x, approximate=True)

# Sample data
x = torch.randn(10)

# This will cause a TypeError
try:
    result = gelu(x)
    print(f"Result: {result}")
except TypeError as e:
    print(f"Error: {e}")