import torch

def get_magnitude(level, level_std):
    # Incorrectly using fixed std of 1 instead of level_std
    return level + torch.randn(1)  # Missing level_std multiplier

# Test with varying level and level_std
levels = torch.tensor([0.5, 1.0, 1.5])
level_std = 0.5
magnitudes = [get_magnitude(l, level_std) for l in levels]
print("Magnitudes:", magnitudes)

import torch

def get_magnitude(level, level_std):
    # Incorrectly using fixed std of 1 instead of level_std
    return level + torch.randn(1)  # Missing level_std multiplier

# Test with varying level and level_std
levels = torch.tensor([0.5, 1.0, 1.5])
level_std = 0.5
magnitudes = [get_magnitude(l, level_std) for l in levels]
print("Magnitudes:", magnitudes)