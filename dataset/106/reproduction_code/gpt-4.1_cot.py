import torch
from torch.cdist import cdist

# Step 1: Create two large tensors with large values to trigger float precision issues
# Using float32 to mimic the original scenario, large values can cause (x-y)^2 computed as x^2 + y^2 - 2xy
# to become slightly negative due to floating point errors

# Create tensor x with large values
x = torch.full((5, 3), 1e10, dtype=torch.float32)
# Create tensor y with large values, slightly different from x
y = torch.full((6, 3), 1e10 + 1, dtype=torch.float32)

# Step 2: Compute cdist normally (this uses the formula internally)
dist = cdist(x, y, p=2)

print("Distance matrix without clamp:")
print(dist)

# Step 3: Manually compute squared Euclidean distance as (x^2 + y^2 - 2xy)
# to demonstrate the negative values due to floating point error

x_sq = (x**2).sum(dim=1, keepdim=True)  # shape (5,1)
y_sq = (y**2).sum(dim=1, keepdim=True).T  # shape (1,6)
xy = x @ y.T  # shape (5,6)

# Compute squared distances
sq_dist = x_sq + y_sq - 2 * xy  # may contain tiny negative values due to precision

print("\nSquared distances (may contain negatives):")
print(sq_dist)

print("\nAny negatives in squared distances?")
print((sq_dist < 0).any().item())

# Step 4: Clamp negative values to zero before sqrt to avoid NaNs
clamped_sq_dist = torch.clamp(sq_dist, min=0)
dist_fixed = torch.sqrt(clamped_sq_dist)

print("\nDistances after clamping squared distances to min=0:")
print(dist_fixed)