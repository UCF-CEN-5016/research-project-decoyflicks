import torch
from vit_pytorch.cross_vit import CrossViT

# Create model with incompatible dimensions
model = CrossViT(
    image_size=256,
    num_classes=10,
    depth=4,                  # Number of transformer blocks
    sm_dim=192,               # Small patch dimension
    sm_patch_size=16,         # Small patch size
    sm_enc_depth=2,           # Small patch encoder depth
    sm_enc_heads=8,           # Small patch encoder heads
    sm_enc_mlp_dim=2048,      # Small patch MLP dimension
    lg_dim=384,               # Large patch dimension
    lg_patch_size=64,         # Large patch size
    lg_enc_depth=3,           # Large patch encoder depth
    lg_enc_heads=8,           # Large patch encoder heads
    lg_enc_mlp_dim=2048,      # Large patch MLP dimension
    cross_attn_depth=2,       # Cross attention depth
    cross_attn_heads=8,       # Cross attention heads
    dropout=0.1,
    emb_dropout=0.1
)

# Input tensor with incorrect channel dimensions
x = torch.randn(1, 3, 256, 256)  # (batch, channels, height, width)

# This will raise dimension mismatch error
out = model(x)
print(out.shape)