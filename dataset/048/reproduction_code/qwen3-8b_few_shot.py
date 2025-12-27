import torch

# Create tensors with mismatched sizes along dim=1
tensor1 = torch.randn(1, 1649)  # First tensor with size 1649
tensor2 = torch.randn(1, 1799)  # Second tensor with size 1799
emissions_arr = [tensor1, tensor2]

# Attempt to concatenate tensors along dim=1
emissions = torch.cat(emissions_arr, dim=1).squeeze()

# This will raise RuntimeError: Sizes of tensors must match except in dimension 1