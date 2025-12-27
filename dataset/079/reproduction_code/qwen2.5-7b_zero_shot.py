import torch
import torch.nn as nn

class RegionViTLayerNorm(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.norm = nn.LayerNorm(input_shape[2:])  # Normalize over spatial dimensions (H, W)

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