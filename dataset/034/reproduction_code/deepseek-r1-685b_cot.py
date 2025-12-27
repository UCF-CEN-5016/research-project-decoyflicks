import torch
import torch.nn.functional as F

# Original problematic call (would trigger the error)
try:
    x = torch.randn(3, 4)
    result = F.gelu(x, approximate=True)  # This raises TypeError
except TypeError as e:
    print(f"Error: {e}")

# Corrected version
x = torch.randn(3, 4)
result = F.gelu(x, approximate='tanh')  # Works with string parameter
print("Successfully executed with string parameter")