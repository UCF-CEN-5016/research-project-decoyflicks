import torch

# Define two tensors
x = torch.tensor([1.0], dtype=torch.float32)
y = torch.tensor([1.0], dtype=torch.float32)

# Compute squared Euclidean distance
distance = torch.dist(x, y, p=2).pow(2)

# Introduce a small error to simulate a negative value (hypothetically)
error = torch.tensor([-1e-10], dtype=torch.float32)
distance_with_error = distance + error

# Apply torch.clamp to ensure the result is at least zero
clamped_distance = torch.clamp(distance_with_error, min=0.0)

print("Original distance:", distance.item())
print("Distance with error:", distance_with_error.item())
print("Clamped distance:", clamped_distance.item())