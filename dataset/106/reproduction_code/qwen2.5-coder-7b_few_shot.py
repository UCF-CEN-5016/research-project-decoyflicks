import numpy as np

def compute_squared_distance(x, y)
    return np.sum(np.square(x) + np.square(y) - 2 * x * y)

# Use float16 for limited precision
x = np.array([1e10], dtype=np.float16)
y = np.array([1e10 + 1], dtype=np.float16)

squared_distance = compute_squared_distance(x, y)
print(squared_distance)