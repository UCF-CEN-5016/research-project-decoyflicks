import torch
import torch.nn.functional as F

# Define a custom gelu function with the incorrect approximate parameter type
def gelu(x):
    return F.gelu(x, approximate=True)

# Create a sample tensor
x = torch.randn(1, 10)

# Attempt to apply the custom gelu function to the tensor
try:
    result = gelu(x)
except TypeError as e:
    print(f"Error: {e}")

# Corrected gelu function with the approximate parameter as a string
def corrected_gelu(x):
    return F.gelu(x, approximate='approximate')

# Apply the corrected gelu function to the tensor
result = corrected_gelu(x)
print(f"Result shape: {result.shape}")