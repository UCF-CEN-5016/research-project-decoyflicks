import torch
from vit_pytorch.cross_vit import CrossViT

batch_size = 32
image_size = 256
num_classes = 1000

# Create input data
input_data = torch.randn(batch_size, 3, image_size, image_size)

# Initialize CrossViT with minimal parameters to trigger the bug
try:
    model = CrossViT(
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
        cross_attn_heads=8
    )
    print("CrossViT initialized successfully")
except Exception as e:
    print(f"CrossViT initialization error: {e}")
    
    # For debugging purposes, let's try a simpler initialization
    try:
        model_simple = CrossViT(
            image_size=image_size,
            num_classes=num_classes
        )
        print("Simple CrossViT initialized successfully")
    except Exception as e:
        print(f"Simple CrossViT initialization error: {e}")

# If model initialization succeeded, try forward pass
if 'model' in locals():
    try:
        predictions = model(input_data)
        print(f"Forward pass successful. Output shape: {predictions.shape}")
    except Exception as e:
        print(f"Forward pass error: {e}")