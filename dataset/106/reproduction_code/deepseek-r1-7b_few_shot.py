import torch

# Ensure we are using half precision to trigger the problem
x = torch.ones(32, 10, dtype=torch.float16)
delta = torch.randn(32, 10) * 1e-5  # Small perturbations
y = x + delta

distance_matrix = torch.cdist(x, y)

print("Distance matrix max:", distance_matrix.max())
print("Any negative values?", (distance_matrix < 0).any())

import torch

x = torch.randn(32, 10).half()  # Using half-precision for lower values
y = x + (torch.randn(32, 10) * 5e-4).half()  # Small perturbation

# Compute difference using the problematic method
difference_squared = (x**2 + y**2 - 2 * torch.mul(x, y))

# Take sqrt to get distance
distance = torch.sqrt(difference_squared)

print("Maximum value in difference_squared:", difference_squared.max())

import numpy as np

def test_half_precision_underflow():
    # Create two arrays with small values that will cause issues due to half-precision limitations.
    x = np.full((4,), 1e-8, dtype=np.float16)
    y = np.array([0.5, 0.3, 1e-9, 2e-9], dtype=np.float16)
    
    # Compute difference squared using the formula that can lead to negative values due to precision issues.
    x_squared = x ** 2
    y_squared = y ** 2
    product_term = np.multiply(x, y) * 2
    
    difference_squared = x_squared + y_squared - product_term
    
    print("x:", x)
    print("y:", y)
    print("x squared:", x_squared)
    print("y squared:", y_squared)
    print("Product term (2xy):", product_term)
    print("Computed difference_squared:", difference_squared)

test_half_precision_underflow()