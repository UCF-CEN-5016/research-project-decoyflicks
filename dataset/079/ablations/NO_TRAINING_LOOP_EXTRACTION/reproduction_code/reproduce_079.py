import torch
from torch import nn
from einops import rearrange
from vit_pytorch.rvt import RvT

# Setup
batch_size = 2
image_size = 32
channels = 3
patch_size = 4
num_classes = 10
dim = 128
depth = 6
heads = 8
mlp_dim = 256

# Create random input data
input_data = torch.randn(batch_size, channels, image_size, image_size)

# Instantiate the RvT class
model = RvT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, channels=channels)

# Modify FeedForward class for incorrect LayerNorm dimensions
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., use_glu=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim, dim=1),  # Incorrect dimension
            nn.Linear(dim, hidden_dim * 2 if use_glu else hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Run the forward method
output = model(input_data)