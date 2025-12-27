import torch
from torch.nn import LayerNorm

# Minimal setup for RegionViT
class MinimalRegionViT(torch.nn.Module):
    def __init__(self):
        super(MinimalRegionViT, self).__init__()
        self.layernorm = LayerNorm(128)  # Set the number of channels (C)

    def forward(self, x):
        return self.layernorm(x)

# Create a sample tensor
input_tensor = torch.randn(1, 128, 224, 224)  # (N, C, H, W)

# Create an instance of the MinimalRegionViT model
model = MinimalRegionViT()

# Run the forward pass
output = model(input_tensor)

import torch
from torch.nn import LayerNorm

class MinimalRegionViT(torch.nn.Module):
    def __init__(self):
        super(MinimalRegionViT, self).__init__()
        self.layernorm = LayerNorm(128)  # Set the number of channels (C)

    def forward(self, x):
        return self.layernorm(x)

# Create a sample tensor
input_tensor = torch.randn(1, 128, 224, 224)  # (N, C, H, W)

# Create an instance of the MinimalRegionViT model
model = MinimalRegionViT()

# Run the forward pass
output = model(input_tensor)