import torch
from vit_pytorch import ViT
from vit_pytorch.mpp import MPP

# Minimal ViT model configuration
model = ViT(
    image_size=256,
    patch_size=32,
    num_classes=1000,
    dim=1024,
    depth=2,  # Reduced depth for minimal reproduction
    heads=2,  # Fewer heads
    mlp_dim=2048,
)

# MPP trainer with same config as model
mpp_trainer = MPP(
    transformer=model,
    patch_size=32,
    dim=1024,
    mask_prob=0.15,
    random_patch_prob=0.30,
    replace_prob=0.50,
)

# Sample smaller batch to make error clearer
def sample_images():
    return torch.randn(2, 3, 256, 256)  # Smaller batch size

# This will trigger the dimension error
images = sample_images()
try:
    loss = mpp_trainer(images)
except RuntimeError as e:
    print("Error occurred:")
    print(e)
    print("\nThe issue occurs because:")
    print("1. The patch embeddings produce [batch, num_patches, patch_dim]")
    print("2. But layer_norm expects last dimension to match model dim (1024)")
    print("3. Actual patch_dim is 3*32*32=3072 which doesn't match model dim")