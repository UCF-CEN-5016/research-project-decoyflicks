import torch

# Minimal setup: create a tensor using torch.Tensor
uninitialized_tensor = torch.Tensor(100, 100)

# Triggering conditions: sum the tensor to observe the behavior
print("Sum of uninitialized tensor:", uninitialized_tensor.sum())

# Compare with torch.zeros
initialized_tensor = torch.zeros(100, 100)
print("Sum of initialized tensor:", initialized_tensor.sum())

# Observe the difference
print("Difference between uninitialized and initialized tensors:")
print(uninitialized_tensor.sum() - initialized_tensor.sum())