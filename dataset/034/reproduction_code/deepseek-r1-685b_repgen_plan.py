import torch

def gelu(x):
    return torch.nn.functional.gelu(x, approximate='tanh')  # Use 'tanh' as string argument

# Sample input tensor
x = torch.randn(10)

# Handle the TypeError if occurs
try:
    output = gelu(x)
    print("GELU output:", output)
except TypeError as e:
    print("Error:", e)
    print("Solution: Use 'tanh' as string argument instead of boolean")