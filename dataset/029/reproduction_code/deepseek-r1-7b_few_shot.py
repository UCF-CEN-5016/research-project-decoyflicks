import torch

# Attempt to check MPS availability without it being set
if torch.backends.mps.is_available():
    print("MPS is available")
else:
    print("MPS is not available")

# Then proceed with the code that uses it, which will cause an error if not handled

import torch

# Set device as mps if available else cuda:0 or something
device = 'mps' if torch.backends.mps.is_available() else 'cuda'

# Initialize model
model = torch.nn.Sequential(
    torch.nn.Linear(1, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 1)
)

# Data (simplified to trigger error)
x = torch.randn(32, 1)

try:
    # Something that would use MPS but isn't available
    y = model(x)
except AttributeError as e:
    print(f"AttributeError: {e}")

import torch

print(torch.backends.mps)  # This line will raise AttributeError because 'mps' is not an attribute of 'torch.backends'

import torch

# This line raises an AttributeError since 'mps' is not an attribute of 'torch.backends'
print(torch.backends.mps)