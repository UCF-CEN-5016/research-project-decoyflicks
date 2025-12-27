import torch

# Incorrect GELU implementation that triggers the bug
def faulty_gelu(x):
    return torch.nn.functional.gelu(x, approximate=True)  # Should be string like 'tanh'

# Test case that reproduces the error
x = torch.randn(5)  # Random input tensor

# This will raise: TypeError: gelu(): argument 'approximate' must be str, not bool
output = faulty_gelu(x)
print(output)