import torch
from vit_pytorch.cross_vit import CrossViT
import matplotlib.pyplot as plt
import numpy as np

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

# Test with different image types
def test_model_with_image(channels, name):
    try:
        # Create image with specified number of channels
        img = torch.randn(1, channels, 256, 256)
        
        # Try to process with model
        output = model(img)
        return f"{name} (channels={channels}): Success"
    except Exception as e:
        return f"{name} (channels={channels}): Failed - {str(e)}"

# Test with RGB image (should work)
print(test_model_with_image(3, "RGB"))

# Test with grayscale image (should fail - this reproduces the bug)
print(test_model_with_image(1, "Grayscale"))

# Test with RGBA image (might work)
print(test_model_with_image(4, "RGBA"))