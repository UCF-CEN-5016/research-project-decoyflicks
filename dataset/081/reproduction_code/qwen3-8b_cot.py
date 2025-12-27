import torch
import torch.nn as nn
from einops import rearrange

# Define a minimal model that replicates the bug
class ImageEmbedder(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super(ImageEmbedder, self).__init__()
        self.patch_size = patch_size
        self.to_embedding = nn.Linear(patch_size * patch_size * 3, embed_dim)  # <-- This line is problematic for 1-channel input

    def forward(self, img):
        # Original code that assumes 3 channels and doesn't generalize
        img = rearrange(img, 'b c (h h0) (w w0) -> b (h w) (h0 w0 c)', h0=self.patch_size, w0=self.patch_size)
        img = self.to_embedding(img)
        return img

# Reproduction code
def reproduce_bug():
    # Setup a model with 3-channel expectation
    model = ImageEmbedder(patch_size=16, embed_dim=768)

    # Create a 1-channel input tensor (batch size 1, 1 channel, 256x256)
    input_tensor = torch.randn(1, 1, 256, 256)

    # Trigger the bug
    with torch.no_grad():
        output = model(input_tensor)
        print(f"Output shape: {output.shape}")

# Run the reproduction
reproduce_bug()

self.to_embedding = nn.Linear(patch_size * patch_size * input_channels, embed_dim)

class ImageEmbedder(nn.Module):
    def __init__(self, patch_size, embed_dim, input_channels=3):
        super(ImageEmbedder, self).__init__()
        self.patch_size = patch_size
        self.to_embedding = nn.Linear(patch_size * patch_size * input_channels, embed_dim)

    def forward(self, img):
        img = rearrange(img, 'b c (h h0) (w w0) -> b (h w) (h0 w0 c)', h0=self.patch_size, w0=self.patch_size)
        img = self.to_embedding(img)
        return img