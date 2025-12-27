import torch

# Create two sets of points with large values to trigger floating point precision issues
x = torch.tensor([[1e10, 1e10]], dtype=torch.float32)
y = torch.tensor([[1e10 + 1, 1e10 + 1]], dtype=torch.float32)

# Manually compute squared distances using the unstable formula
# (x - y)^2 = x^2 + y^2 - 2xy
x_norm = (x ** 2).sum(dim=1, keepdim=True)
y_norm = (y ** 2).sum(dim=1, keepdim=True)
dist_squared = x_norm + y_norm.T - 2 * x @ y.T

print("Squared distances before clamp:", dist_squared)

# Without clamp, sqrt can produce NaN if dist_squared is slightly negative
dist = torch.sqrt(dist_squared)
print("Distances without clamp:", dist)

# Apply clamp to fix negative values
dist_clamped = torch.sqrt(torch.clamp(dist_squared, min=0))
print("Distances with clamp:", dist_clamped)