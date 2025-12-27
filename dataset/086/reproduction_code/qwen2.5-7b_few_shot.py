import torch
import einops

# Simulate the problematic tensor shape [1, 32]
power = torch.arange(32).unsqueeze(0)  # Shape: [1, 32]

# Attempt to reshape using a pattern expecting 1D input
try:
    rearranged = einops.rearrange(power, 'n -> n 1')
    print("Reshaped tensor:", rearranged.shape)
except Exception as e:
    print("Error:", e)