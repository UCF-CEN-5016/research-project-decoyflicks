import torch

# Create two tensors with different sizes along dim=0 but same size along dim=1
tensor1 = torch.randn(1649, 10)
tensor2 = torch.randn(1799, 10)

# Attempt to concatenate along dim=1 (columns)
# This will raise RuntimeError because dim=0 sizes differ (1649 vs 1799)
try:
    result = torch.cat([tensor1, tensor2], dim=1)
except RuntimeError as e:
    print(f"Caught RuntimeError: {e}")