import torch
from vit_pytorch.cross_vit import CrossViT

v = CrossViT(
    image_size=256,
    num_classes=1000,
    depth=4,
    sm_dim=192,
    sm_patch_size=16,
    lg_dim=384,
    lg_patch_size=64,
    cross_attn_depth=2,
    cross_attn_heads=8,
    dropout=0.1,
    emb_dropout=0.1
)

img = torch.randn(1, 1, 256, 256)  # Using 1 channel

try:
    pred = v(img)
except RuntimeError as e:
    print(e)