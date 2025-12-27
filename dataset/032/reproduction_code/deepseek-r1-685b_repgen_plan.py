import torch

# Correct GELU implementation
def gelu(x):
    return 0.5 * x * (1 + torch.tanh((torch.sqrt(2 / 3.14159265358979323846) * (x + 0.044715 * torch.pow(x, 3)))))

# Test case
x = torch.randn(5)  # Random input tensor
output = gelu(x)
print(output)