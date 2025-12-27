import torch
from vit_pytorch import ViT

model = ViT(
    image_size = 224,
    patch_size = 16,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048
)

# Regular channel-first input for ViT
x = torch.randn(2, 3, 224, 224)  # Standard PyTorch image format

output = model(x)

# Print output shape
print(f"Output shape: {output.shape}")