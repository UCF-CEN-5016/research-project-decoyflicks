import torch
from torch import nn
from einops import rearrange

# Minimal reproduction of the bug
def bug_reproduction():
    # Setup parameters
    C, H, W = 64, 8, 8  # Channels, Height, Width
    batch_size = 2
    
    # Original problematic implementation
    class LocalTokenEmbedding(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.norm = nn.LayerNorm(dim)  # Bug: Normalizing over wrong dimension
            
        def forward(self, x):
            # x shape: (batch, channels, height, width)
            x = rearrange(x, 'b c h w -> b (h w) c')  # Flatten spatial dims
            return self.norm(x)
    
    # Create test input
    x = torch.randn(batch_size, C, H, W)
    
    # Instantiate and run problematic layer
    layer = LocalTokenEmbedding(C)
    try:
        out = layer(x)
        print("Output shape:", out.shape)  # May run but normalization is incorrect
    except Exception as e:
        print("Error encountered:", e)
    
    # Correct implementation for comparison
    class FixedLocalTokenEmbedding(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.norm = nn.LayerNorm([dim, H, W])  # Normalize over channels+spatial
            
        def forward(self, x):
            return self.norm(x)  # Keep original shape
    
    # Test fixed version
    fixed_layer = FixedLocalTokenEmbedding(C)
    try:
        fixed_out = fixed_layer(x)
        print("Fixed output shape:", fixed_out.shape)
    except Exception as e:
        print("Error in fixed version:", e)

bug_reproduction()