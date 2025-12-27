import torch

# Define the gelu() function with approximate parameter
def gelu(x, approximate=True):
    return torch.nn.functional.gelu(x, approximate=True)

# Trigger the bug
print(gelu(torch.tensor([1.0])))