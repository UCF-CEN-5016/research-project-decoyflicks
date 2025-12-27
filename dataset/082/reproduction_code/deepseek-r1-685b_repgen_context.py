import torch
from vit_pytorch import ViT
from vit_pytorch.mpp import MPP

def sample_images(batch_size=2, channels=3, image_size=256):
    return torch.randn(batch_size, channels, image_size, image_size)

def handle_error(exception):
    print("Error occurred:")
    print(exception)
    print("\nThe issue occurs because:")
    print("1. The patch embeddings produce [batch, num_patches, patch_dim]")
    print("2. But layer_norm expects last dimension to match model dim (1024)")
    print("3. Actual patch_dim is 3*32*32=3072 which doesn't match model dim")

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

# MPP trainer with the same config as the model
mpp_trainer = MPP(
    transformer=model,
    patch_size=32,
    dim=1024,
    mask_prob=0.15,
    random_patch_prob=0.30,
    replace_prob=0.50,
)

# Sample smaller batch to make error clearer
images = sample_images()
try:
    loss = mpp_trainer(images)
except RuntimeError as e:
    handle_error(e)