import torch

# Create a tensor of shape (1024, 768) with random values between -5 and 5
x = torch.randn(1024, 768) * 10

# Call the gelu function with the created tensor as argument and approximate parameter set to True
torch.nn.functional.gelu(x, approximate=True)