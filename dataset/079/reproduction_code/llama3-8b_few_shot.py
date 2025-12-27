import torch
from vit_pytorch import RegionViT

# Create a RegionViT model
model = RegionViT(
    image_size=224,
    num_classes=1000,
    dim=128,
    depth=4,
    heads=8,
    mlp_dim=2048,
)

# Use the model for local token embedding
input_tensor = torch.randn(1, 3, 14, 14)  # Sample input tensor

output = model.local_token_embedding(input_tensor)  # Should raise an error

print(output.shape)