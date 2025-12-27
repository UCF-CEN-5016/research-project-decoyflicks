import torch
import torch.nn as nn
from einops import rearrange

class ImageEmbedder(nn.Module):
    def __init__(self, patch_size, embed_dim, input_channels=3):
        super(ImageEmbedder, self).__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.to_embedding = nn.Linear(patch_size * patch_size * input_channels, embed_dim)

    def forward(self, img):
        img = rearrange(img, 'b c (h h0) (w w0) -> b (h w) (h0 w0 c)', h0=self.patch_size, w0=self.patch_size)
        img = self.to_embedding(img)
        return img

def reproduce_bug():
    model = ImageEmbedder(patch_size=16, embed_dim=768)
    input_tensor = torch.randn(1, 1, 256, 256)  # Input tensor with 1 channel
    with torch.no_grad():
        output = model(input_tensor)
        print(f"Output shape: {output.shape}")

reproduce_bug()