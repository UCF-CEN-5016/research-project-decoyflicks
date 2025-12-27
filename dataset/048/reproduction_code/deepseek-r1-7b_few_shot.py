import torch

# Simulate two tensors with different sizes along dimension 1 (features)
tensor1 = torch.randn(2, 1649)  # batch_size=2, feature_dim=1649
tensor2 = torch.randn(2, 1799)

try:
    concatenated_tensor = torch.cat([tensor1, tensor2], dim=1).squeeze()
except RuntimeError as e:
    print(f"RuntimeError: {e}")