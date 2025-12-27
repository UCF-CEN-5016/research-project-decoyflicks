import torch

# Correct GELU implementation with 'tanh' as the activation function
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(0.7978845608 * (x + 0.044715 * torch.pow(x, 3))))

# Test case to use the correct GELU implementation
x = torch.randn(5)  # Random input tensor
output = gelu(x)
print(output)