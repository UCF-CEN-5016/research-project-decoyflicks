import torch

# Create tensors with mismatched sizes along dim=1
tensor1 = torch.randn(1, 1649)  # First tensor with size 1649
tensor2 = torch.randn(1, 1799)  # Second tensor with size 1799

# Concatenate tensors along dim=1 after resizing to match
min_size = min(tensor1.shape[1], tensor2.shape[1])
tensor1 = tensor1[:, :min_size]
tensor2 = tensor2[:, :min_size]
emissions = torch.cat([tensor1, tensor2], dim=0)

# Verify the shape of the concatenated tensor
print(emissions.shape)