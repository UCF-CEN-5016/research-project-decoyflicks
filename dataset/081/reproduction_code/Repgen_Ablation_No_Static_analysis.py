import torch
from vit_pytorch.cross_vit import CrossViT
from torchvision.transforms import Compose, Resize, ToTensor, Grayscale, Lambda

# Initialize CrossViT model
model = CrossViT(
    image_size=256,
    num_classes=1000,
    depth=4,
    sm_dim=192,
    sm_patch_size=16,
    sm_enc_depth=2,
    sm_enc_heads=8,
    sm_enc_mlp_dim=2048,
    lg_dim=384,
    lg_patch_size=64,
    lg_enc_depth=3,
    lg_enc_heads=8,
    lg_enc_mlp_dim=2048,
    cross_attn_depth=2,
    cross_attn_heads=8
)

# Create a single-channel image
grayscale_img = torch.randn(1, 1, 256, 256)

# Convert grayscale to 3-channel by repeating the channel
# This avoids the bug by transforming to expected format
def convert_grayscale_to_rgb(x):
    return x.repeat(1, 3, 1, 1)

# Apply conversion to 3-channel
rgb_img = convert_grayscale_to_rgb(grayscale_img)

# This will work because we're providing a 3-channel image
output = model(rgb_img)
print(f"Prediction shape: {output.shape}")

# Verify that we're using a 3-channel image now
print(f"Input image shape: {rgb_img.shape}")