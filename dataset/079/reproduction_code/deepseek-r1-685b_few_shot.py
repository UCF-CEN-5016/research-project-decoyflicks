import torch
import torch.nn as nn

# Problematic implementation
class ProblematicLocalTokenEmbedding(nn.Module):
    def __init__(self, dim, dim_out, window_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim_out, 1),
            nn.LayerNorm(dim_out),  # Incorrect: expects (N, C) but gets (N, C, H, W)
            nn.GELU()
        )
        self.window_size = window_size

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        return x

# Test case that exposes the issue
def test_layer_norm():
    dim = 64
    dim_out = 128
    window_size = 8
    model = ProblematicLocalTokenEmbedding(dim, dim_out, window_size)
    
    # Input tensor (batch, channels, height, width)
    x = torch.randn(2, dim, 32, 32)
    
    try:
        output = model(x)  # This will raise an error
        print("Output shape:", output.shape)
    except RuntimeError as e:
        print("Error encountered:", e)
        print("Expected behavior: LayerNorm expects (N, C) but got (N, C, H, W)")

test_layer_norm()