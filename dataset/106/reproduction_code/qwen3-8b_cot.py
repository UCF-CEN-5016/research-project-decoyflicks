import torch

# Create two tensors with large values
x = torch.tensor([1e10])
y = torch.tensor([1e10 + 1e-5])

# Compute the squared distance
distance = (x - y) ** 2

# Clamp the distance to avoid negative values due to floating-point errors
distance = torch.clamp(distance, min=0)

print("Squared distance:", distance)