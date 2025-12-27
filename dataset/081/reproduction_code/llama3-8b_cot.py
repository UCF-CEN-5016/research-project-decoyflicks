import torch
from vit_pytorch.cross_vit import CrossViT

# Minimal setup
image_size = 256
num_classes = 1000
depth = 4  # number of multi-scale encoding blocks
sm_dim = 192  # high res dimension
sm_patch_size = 16  # high res patch size (should be smaller than lg_patch_size)
sm_enc_depth = 2  # high res depth
sm_enc_heads = 8  # high res heads
sm_enc_mlp_dim = 2048  # high res feedforward dimension
lg_dim = 384  # low res dimension
lg_patch_size = 64  # low res patch size
lg_enc_depth = 3  # low res depth
lg_enc_heads = 8  # low res heads
lg_enc_mlp_dim = 2048  # low res feedforward dimensions
cross_attn_depth = 2  # cross attention rounds
cross_attn_heads = 8  # cross attention heads
dropout = 0.1
emb_dropout = 0.1

# Triggering conditions: Use an image with more than 3 channels
img = torch.randn(1, 4, 256, 256)  # Using 4 channels

# Initialize the model
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

# Run the model
pred = v(img)  # This should raise a RuntimeError