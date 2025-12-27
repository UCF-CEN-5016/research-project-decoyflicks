import torch
import torch.nn as nn

# Define a minimal RegionViT-like model
class RegionViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = nn.LayerNorm(16)  # Incorrect dimension
        self.conv = nn.Conv2d(16, 16, kernel_size=3)

    def forward(self, x):
        # Simulate local token embedding
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # NHWC format
        x = self.layer_norm(x)  # This will cause an error
        return x

# Create a sample input
input_tensor = torch.randn(1, 16, 32, 32)

# Create an instance of the model
model = RegionViT()

# This will raise an error due to incorrect LayerNorm dimension
try:
    output = model(input_tensor)
    print(output.shape)
except RuntimeError as e:
    print(f"Error: {e}")