import torch

# Create two tensors with the same dimensions
tensor_a = torch.randn(2, 3, 4)  # shape (2, 3, 4)
tensor_b = torch.randn(2, 3, 4)  # shape (2, 3, 4)

# Multiply the tensors element-wise
result = torch.mul(tensor_a, tensor_b)