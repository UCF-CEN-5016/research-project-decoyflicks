import torch

def get_magnitude(level, level_std):
    # Correctly using level_std multiplier
    return level + torch.randn(1) * level_std

# Test with varying level and level_std
levels = torch.tensor([0.5, 1.0, 1.5])
level_std = 0.5
magnitudes = [get_magnitude(l, level_std) for l in levels]
print("Magnitudes:", magnitudes)