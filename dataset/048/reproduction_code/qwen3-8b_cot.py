import torch

# Simulate the problem by creating two tensors with mismatched sizes in a dimension other than dim=1
tensor1 = torch.randn(1, 1649)  # Shape: (batch_size, 1649)
tensor2 = torch.randn(1, 1799)  # Shape: (batch_size, 1799)

# Attempt to concatenate along dim=1 (which is the feature dimension)
try:
    concatenated = torch.cat((tensor1, tensor2), dim=1)
    print("Concatenation successful!")
except RuntimeError as e:
    print("Error occurred:", e)