import torch

# Create two tensors with large values
x = torch.tensor([1e10])
y = torch.tensor([1e10 + 1e-5])

# Compute the squared distance
distance = torch.max((x - y) ** 2, torch.tensor(0))

print("Squared distance:", distance)