import torch

def gelu_correct(x):
    # Correct usage with string parameter
    return torch.nn.functional.gelu(x, approximate='tanh')

# Create a sample input tensor
input_tensor = torch.randn(10, 768)

# This will work correctly
output = gelu_correct(input_tensor)
print(f"Output shape: {output.shape}")