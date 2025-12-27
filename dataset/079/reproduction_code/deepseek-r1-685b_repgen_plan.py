import torch
import torch.nn as nn

class LocalTokenEmbedding(nn.Module):
    def __init__(self, dim, dim_out, window_size):
        super(LocalTokenEmbedding, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim_out, 1),
            nn.LayerNorm(dim_out),
            nn.GELU()
        )
        self.window_size = window_size

    def forward(self, x):
        x = self.proj(x)
        return x

def test_layer_norm():
    dim = 64
    dim_out = 128
    window_size = 8
    model = LocalTokenEmbedding(dim, dim_out, window_size)
    
    x = torch.randn(2, dim, 32, 32)
    
    try:
        output = model(x)
        print("Output shape:", output.shape)
    except RuntimeError as e:
        print("Error encountered:", e)
        print("Expected behavior: LayerNorm expects (N, C) but got (N, C, H, W)")

test_layer_norm()