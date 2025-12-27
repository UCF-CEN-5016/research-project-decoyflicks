import torch

# Create a tensor with uninitialized values
bias = torch.Tensor(100, 100)

# Check the sum to observe potential non-determinism or NaNs
print("Sum of uninitialized bias tensor:", bias.sum())

# Example of potential NaN output
# Note: The result may vary due to uninitialized memory