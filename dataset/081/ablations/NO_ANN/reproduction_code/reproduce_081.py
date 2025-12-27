import torch
from vit_pytorch.cross_vit import CrossViT

image_size = 256
num_classes = 1000
depth = 4
sm_dim = 192
sm_patch_size = 16
sm_enc_depth = 2
sm_enc_heads = 8
sm_enc_mlp_dim = 2048
lg_dim = 384
lg_patch_size = 64
lg_enc_depth = 3
lg_enc_heads = 8
lg_enc_mlp_dim = 2048
cross_attn_depth = 2
cross_attn_heads = 8
dropout = 0.1
emb_dropout = 0.1

img = torch.randn(1, 1, 256, 256)  # Single-channel image

v = CrossViT(
    image_size=image_size,
    num_classes=num_classes,
    depth=depth,
    sm_dim=sm_dim,
    sm_patch_size=sm_patch_size,
    sm_enc_depth=sm_enc_depth,
    sm_enc_heads=sm_enc_heads,
    sm_enc_mlp_dim=sm_enc_mlp_dim,
    lg_dim=lg_dim,
    lg_patch_size=lg_patch_size,
    lg_enc_depth=lg_enc_depth,
    lg_enc_heads=lg_enc_heads,
    lg_enc_mlp_dim=lg_enc_mlp_dim,
    cross_attn_depth=cross_attn_depth,
    cross_attn_heads=cross_attn_heads,
    dropout=dropout,
    emb_dropout=emb_dropout
)

try:
    pred = v(img)
except Exception as e:
    print(e)