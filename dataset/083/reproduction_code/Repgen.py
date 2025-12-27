import torch
from vit_pytorch.cross_vit import CrossViT

# Define comprehensive steps to reproduce the CrossViT initialization bug
def reproduce_crossvit_bug():
    # Step 1: Define model parameters
    image_size = 256
    num_classes = 10
    patch_size = 16
    dim = 768
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
    
    # Step 2: Initialize the CrossViT model with all parameters
    # This will trigger the initialization bug in the CrossViT module
    print("Attempting to initialize CrossViT model...")
    try:
        model = CrossViT(
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
        print("CrossViT model initialized successfully!")
    except Exception as e:
        print(f"Error initializing CrossViT model: {str(e)}")
        raise
    
    # Step 3: Create a sample input image
    img = torch.randn(1, 3, image_size, image_size)
    
    # Step 4: Attempt to perform a forward pass
    try:
        print("Attempting forward pass...")
        preds = model(img)
        print(f"Forward pass successful. Output shape: {preds.shape}")
    except Exception as e:
        print(f"Error during forward pass: {str(e)}")
        raise

if __name__ == "__main__":
    reproduce_crossvit_bug()