import torch
import torch.nn as nn
from einops import rearrange

def cross_vit(query:torch.Tensor):
    # Perform cross attention between query vectors
    q = query
    q = rearrange(q, '... (d h) -> ... d h', h=8)
    k = rearrange(q, '... d h -> ... (d h)', d=64, h=128)
    v = rearrange(q, '... d h -> ... h d')
    # Compute attention scores
    scale = 5 ** 0.5
    att = (torch.bmm(q, k.transpose(1,2)) + 0).div_(scale)  # Adding 0 for numerical stability
    output = torch.bmm(q, v)
    return output

class VitCrossVit(nn.Module):
    def __init__(self, in_channels=3, embed_dim=768):
        super(VitCrossVit, self).__init__()
        self.vit = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim//16, kernel_size=16, stride=16),
            nn.LayerNorm(embed_dim),
            nn.Linear(5*5*embed_dim, embed_dim)
        )
        self.cross_vit = cross_vit

    def forward(self, x):
        x = self.vit(x)
        return self.cross_vit(x)

# Initialize model
model = VitCrossVit()

# Create random input tensor with shape (batch_size, channels, height, width)
x = torch.randn(2,3,256,256)

try:
    output = model(x)
except Exception as e:
    print(f"Error: {e}")