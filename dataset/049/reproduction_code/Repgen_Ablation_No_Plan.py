import torch

# Initialize tensors with zeros
u = torch.zeros(1, 1)
v = torch.zeros(10, 10)

print(u.sum())
print(v.sum())

# Create a tensor and apply operations that might cause NaNs due to uninitialized values
tensor = torch.tensor([[float('nan'), float('nan')], [float('nan'), float('nan')]])
result = tensor.sum()
print(result)