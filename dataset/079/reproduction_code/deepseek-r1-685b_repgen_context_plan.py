import torch
import torch.nn as nn

class LocalTokenEmbedding(nn.Module):
    def __init__(self, dim, dim_out):
        super(LocalTokenEmbedding, self).__init__()
        self.conv = nn.Conv2d(dim, dim_out, 1)
        self.layer_norm = nn.LayerNorm(dim_out)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.gelu(x)
        return x

def test_layer_norm():
    dim = 64
    dim_out = 128
    model = LocalTokenEmbedding(dim, dim_out)
    
    x = torch.randn(2, dim, 32, 32)
    
    output = model(x)
    print("Output shape:", output.shape)

test_layer_norm()