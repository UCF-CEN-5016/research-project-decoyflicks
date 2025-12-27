import torch
import torch.nn.functional as F

# Define a tensor
x = torch.randn(1, 10)

# Incorrect usage of gelu function
try:
    result = F.gelu(x, approximate=True)
except TypeError as e:
    print(f"Error: {e}")

# Correct usage of gelu function
result = F.gelu(x, approximate="tanh")
print(f"Result: {result}")