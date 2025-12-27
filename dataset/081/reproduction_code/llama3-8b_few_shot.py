import torch
from vit_pytorch.cross_vit import CrossViT

v = CrossViT(
    image_size=256,
    num_classes=1000,
    depth=4,  # number of multi-scale encoding blocks
    sm_dim=192,  # high res dimension
    sm_patch_size=16,  # high res patch size (should be smaller than lg_patch_size)
    sm_enc_depth=2,  # high res depth
    sm_enc_heads=8,  # high res heads
    sm_enc_mlp_dim=2048,  # high res feedforward dimension
    lg_dim=384,  # low res dimension
    lg_patch_size=64,  # low res patch size
    lg_enc_depth=3,  # low res depth
    lg_enc_heads=8,  # low res heads
    lg_enc_mlp_dim=2048,  # low res feedforward dimensions
    cross_attn_depth=2,  # cross attention rounds
    cross_attn_heads=8,  # cross attention heads
    dropout=0.1,
    emb_dropout=0.1
)

img = torch.randn(1, 4, 256, 256)  # Using 4 channels
try:
    pred = v(img)
except RuntimeError as e:
    print(f"Error: {e}")