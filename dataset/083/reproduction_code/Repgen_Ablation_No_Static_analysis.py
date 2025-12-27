import torch
from vit_pytorch.cross_vit import CrossViT

def test():
    # Initialize CrossViT with parameters that trigger the initialization bug
    try:
        v = CrossViT(
            image_size=256,
            num_classes=1000,
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
            cross_attn_heads=8,
            dropout=0.1,
            emb_dropout=0.1
        )
        print("CrossViT initialized successfully")
    except Exception as e:
        print(f"CrossViT initialization error: {e}")
        raise

    # Create a test image
    img = torch.randn(1, 3, 256, 256)

    # Try to run a forward pass
    try:
        preds = v(img)
        assert preds.shape == (1, 1000), 'correct logits outputted'
        print("Forward pass successful")
    except Exception as e:
        print(f"Forward pass error: {e}")
        raise

if __name__ == "__main__":
    test()