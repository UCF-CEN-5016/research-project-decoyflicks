import torch

# Define a mock gelu function that expects a string for 'approximate'
def gelu(x, approximate):
    # This is a mock function to simulate the bug
    # In real code, this would be torch.nn.functional.gelu
    if not isinstance(approximate, str):
        raise TypeError("gelu(): argument 'approximate' must be str, not bool")

# Create a random tensor
x = torch.randn(10)

# Call the function with approximate=True (a bool), which triggers the error
gelu(x, approximate=True)