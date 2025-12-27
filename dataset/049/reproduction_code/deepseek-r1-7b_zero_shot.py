import torch

# Uninitialized bias with random values (problematic)
u = torch.Tensor(10, 10)
print("Sum of u:", u.sum())  # May print something like tensor(4.9312e-38) or nan

# Properly initialized bias with zeros
v = torch.zeros(10, 10)
print("Sum of v:", v.sum())