import torch
from vit_pytorch.cross_vit import CrossViT

def initialize_crossvit_model(image_size, num_classes):
    v = CrossViT(
        image_size=image_size,
        num_classes=num_classes,
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
        cross_attn_heads=8,
        dropout=0.1,
        emb_dropout=0.1
    )
    return v

def create_single_channel_image(image_size):
    img = torch.randn(1, 1, image_size, image_size)  # (batch, channels, height, width)
    return img

# Initialize CrossViT model (expecting 3-channel images)
v = initialize_crossvit_model(image_size=256, num_classes=1000)

# Create 1-channel image (instead of 3-channel)
img = create_single_channel_image(image_size=256)

# Attempt to process image (will raise error)
pred = v(img)  # (1, 1000)