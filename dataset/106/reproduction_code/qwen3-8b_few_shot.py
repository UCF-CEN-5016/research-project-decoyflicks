import numpy as np

# Use float16 for limited precision
x = np.array([1e10], dtype=np.float16)
y = np.array([1e10 + 1], dtype=np.float16)

# Compute squared distance
squared_distance = x**2 + y**2 - 2 * x * y
print(squared_distance)