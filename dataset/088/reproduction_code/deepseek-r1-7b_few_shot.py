import torch
from torch.nn.functional import pad

# Create a sample tensor
x = torch.randn(5)

# Apply F.pad without specifying the pad parameter, using value instead
result = pad(x, (0,), mode='constant', value=0)  # Uses zeros on the right side

print("Original tensor:", x)
print("Padded tensor with zeros on the right:\n", result)