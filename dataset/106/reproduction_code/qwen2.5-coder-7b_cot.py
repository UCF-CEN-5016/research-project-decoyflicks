import torch
from vector_quantize_pytorch import cdist

# Set random seed for reproducibility
torch.manual_seed(42)

# Create two vectors with elements set up to trigger the bug
x = torch.tensor([1e8, 0.5], dtype=torch.float16)
y = torch.tensor([1e8 - 5e7, 0], dtype=torch.float16)

# Compute distances which may result in NaN due to floating-point errors
dist = cdist(x.unsqueeze(1), y.unsqueeze(0))

print("Distance matrix with possible NaNs:"
print(dist)