import torch

# Correct GELU implementation
def gelu(x):
    return 0.5 * x * (1 + torch.tanh((math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))))

# Test case
x = torch.randn(5)  # Random input tensor
output = gelu(x)
print(output)