import torch
from vit_pytorch.cross_vit import CrossViT

# Minimal model setup
model = CrossViT(
    image_size=256,
    num_classes=1000,
    depth=4,                  # number of multi-scale encoding blocks
    sm_dim=192,               # high res dimension
    sm_patch_size=16,         # high res patch size (should divide image size)
    sm_enc_depth=2,           # high res depth
    sm_enc_heads=8,           # high res heads
    sm_enc_mlp_dim=2048,      # high res mlp dimension
    lg_dim=384,               # low res dimension
    lg_patch_size=64,         # low res patch size
    lg_enc_depth=3,           # low res depth
    lg_enc_heads=8,           # low res heads
    lg_enc_mlp_dim=2048,      # low res mlp dimension
    cross_attn_depth=2,       # cross attention rounds
    cross_attn_heads=8,       # cross attention heads
    dropout=0.1,
    emb_dropout=0.1
)

# Create input tensor with incorrect dimensions to trigger the bug
x = torch.randn(1, 3, 256, 256)  # Standard image input

# This forward pass will likely trigger the dimension mismatch error
out = model(x)