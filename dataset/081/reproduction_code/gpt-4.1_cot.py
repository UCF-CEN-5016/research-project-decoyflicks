import torch
from vit_pytorch.cross_vit import CrossViT

# Instantiate CrossViT with default parameters
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
    cross_attn_heads=8,
    dropout=0.1,
    emb_dropout=0.1
)

# Create a batch with 1-channel images (instead of 3)
img = torch.randn(1, 1, 256, 256)

# This will raise an error due to channel mismatch in patch embedding convolution
output = model(img)