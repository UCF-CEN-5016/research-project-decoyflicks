norm = nn.LayerNorm([C, H, W])

norm = nn.LayerNorm([H, W])

import torch
import torch.nn as nn

# Define a simple model with an incorrect LayerNorm
class RegionViTLayerNorm(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        # Incorrect: normalize only over spatial dimensions (H, W)
        self.norm = nn.LayerNorm(list(input_shape[1:]))  # [H, W]

    def forward(self, x):
        return self.norm(x)

# Test input: [B, C, H, W] = [1, 3, 2, 2]
x = torch.randn(1, 3, 2, 2)

# Instantiate the model
model = RegionViTLayerNorm(x.shape)

# Forward pass
output = model(x)

# Print the output shape and a sample
print("Output shape:", output.shape)
print("Sample output tensor:\n", output)

self.norm = nn.LayerNorm([C, H, W])

self.norm = nn.LayerNorm([H, W, C])