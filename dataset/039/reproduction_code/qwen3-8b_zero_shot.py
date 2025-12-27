import torch

# Create two tensors with mismatched dimensions in the third dimension (0-based)
tensor_a = torch.randn(2, 3, 4)  # shape (2, 3, 4)
tensor_b = torch.randn(2, 3, 5)  # shape (2, 3, 5)

# Attempt to multiply them
result = tensor_a * tensor_b  # This will raise an error