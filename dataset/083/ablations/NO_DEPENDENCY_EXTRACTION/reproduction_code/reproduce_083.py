import torch
import torch.nn as nn
from vit_pytorch.cross_vit import CrossViT

torch.manual_seed(42)

image_size = 224
num_classes = 10
sm_dim = 64
lg_dim = 128
sm_patch_size = 12
lg_patch_size = 16
sm_enc_depth = 1
lg_enc_depth = 4
sm_enc_heads = 8
lg_enc_heads = 8
cross_attn_heads = 8
sm_enc_mlp_dim = 2048
lg_enc_mlp_dim = 2048
sm_enc_dim_head = 64
lg_enc_dim_head = 64
cross_attn_dim_head = 64
dropout = 0.1
emb_dropout = 0.1

model = CrossViT(
    image_size=image_size,
    num_classes=num_classes,
    sm_dim=sm_dim,
    lg_dim=lg_dim,
    sm_patch_size=sm_patch_size,
    sm_enc_depth=sm_enc_depth,
    sm_enc_heads=sm_enc_heads,
    sm_enc_mlp_dim=sm_enc_mlp_dim,
    sm_enc_dim_head=sm_enc_dim_head,
    lg_patch_size=lg_patch_size,
    lg_enc_depth=lg_enc_depth,
    lg_enc_heads=lg_enc_heads,
    lg_enc_mlp_dim=lg_enc_mlp_dim,
    lg_enc_dim_head=lg_enc_dim_head,
    cross_attn_depth=2,
    cross_attn_heads=cross_attn_heads,
    cross_attn_dim_head=cross_attn_dim_head,
    depth=3,
    dropout=dropout,
    emb_dropout=emb_dropout
)

input_tensor = torch.randn(8, 3, image_size, image_size)
output_logits = model(input_tensor)

assert output_logits.shape == (8, num_classes)
assert not torch.isnan(output_logits).any()

print("Output logits:", output_logits)