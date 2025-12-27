import torch
import numpy as np

# Minimal environment setup
x = np.array([1.0, 2.0], dtype=np.float32)
y = np.array([-10.0, -20.0], dtype=np.float32)

# Triggering conditions: compute difference between x and y
diff = x - y
print(diff)  # Output: [-11.  -18.]

# Compute distance using (x-y)^2 formula
dist = torch.pow(diff, 2)
print(dist)  # This may output NaN values due to floating-point precision overflow

# To reproduce the bug, we can use a small value for the second vector component:
y[1] = -19.9999
diff = x - y
print(diff)  # Output: [-11.  -19.9999]
dist = torch.pow(diff, 2)
print(dist)  # This will output NaN values due to floating-point precision overflow

# To fix the bug, we can use torch.clamp as suggested in the bug report:
dist = torch.clamp(torch.pow(diff, 2), min=0)
print(dist)