import torch
import torch.nn as nn
import einops

from vit_pytorch import (
    ViT,
    Block,
    PatchEmbed,
    CrossAttention,
)

# Ensure CUDA is available; else, fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model Parameters
img_size = 256  # Height/Width of images
patch_size = 16
n_blocks = 4
n_heads = 8
d_model = img_size * img_size // 2  # Reduced dimension for demonstration

# Create dummy input; e.g., batch size=2, channels=3 (RGB), height=width=256
x = torch.randn(2, 3, img_size, img_size).to(device)

class SimpleCrossVit(nn.Module):
    def __init__(self, d_model, n_blocks, n_heads):
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels=3, patch_size=patch_size)
        self.positional编码 is not directly related to the cross-attention bug but might be part of a complete model.
        # For this example, focus on cross-attention implementation
        self.blocks = nn.ModuleList([
            Block(
                dim=d_model,
                n_heads=n_heads,
                cross_attention=True  # Trigger cross-attention layers
            )
            for _ in range(n_blocks)
        ])
    
    def forward(self, x):
        x = self.patch_embed(x)  # Shape: (batch_size, d_model, H, W)
        for block in self.blocks:
            x = block(x)  # Apply cross-attention blocks
        return x

# Initialize model and move to device
model = SimpleCrossVit(d_model=d_model, n_blocks=n_blocks, n_heads=n_heads).to(device)

# Wrap with an optimizer (for demo; actual training would follow)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Forward pass attempt to trigger the bug
try:
    model(x)  # This should trigger cross-attention computation and potentially the bug
except Exception as e:
    print(f"Bug triggered: {e}")