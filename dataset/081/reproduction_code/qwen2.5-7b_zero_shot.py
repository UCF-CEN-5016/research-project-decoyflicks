import torch
from vit_pytorch.cross_vit import CrossViT

# Define model configuration
model_config = {
    'image_size': 256,
    'num_classes': 1000,
    'depth': 4,
    'sm_dim': 192,
    'sm_patch_size': 16,
    'sm_enc_depth': 2,
    'sm_enc_heads': 8,
    'sm_enc_mlp_dim': 2048,
    'lg_dim': 384,
    'lg_patch_size': 64,
    'lg_enc_depth': 3,
    'lg_enc_heads': 8,
    'lg_enc_mlp_dim': 2048,
    'cross_attn_depth': 2,
    'cross_attn_heads': 8,
    'dropout': 0.1,
    'emb_dropout': 0.1
}

# Initialize the model
v = CrossViT(**model_config)

# Generate random input image
img = torch.randn(1, 1, model_config['image_size'], model_config['image_size'])

# Forward pass
pred = v(img)  # (1, 1000)