import torch
from vit_pytorch.cross_vit import CrossViT

class CrossViTConfig:
    """
    Configuration class for CrossViT model parameters.
    """
    def __init__(self):
        self.image_size = 256
        self.num_classes = 10
        self.patch_size = 16 # This parameter is not directly used in the CrossViT constructor but is common for ViT models
        self.dim = 768
        self.depth = 4
        self.sm_dim = 192
        self.sm_patch_size = 16
        self.sm_enc_depth = 2
        self.sm_enc_heads = 8
        self.sm_enc_mlp_dim = 2048
        self.lg_dim = 384
        self.lg_patch_size = 64
        self.lg_enc_depth = 3
        self.lg_enc_heads = 8
        self.lg_enc_mlp_dim = 2048
        self.cross_attn_depth = 2
        self.cross_attn_heads = 8
        self.dropout = 0.1
        self.emb_dropout = 0.1

def initialize_and_test_crossvit(config: CrossViTConfig):
    """
    Initializes the CrossViT model with the given configuration and performs a dummy forward pass.
    This function aims to reproduce and demonstrate any initialization or forward pass bugs.

    Args:
        config (CrossViTConfig): Configuration object containing model parameters.
    """
    print("Attempting to initialize CrossViT model...")
    try:
        model = CrossViT(
            image_size=config.image_size,
            num_classes=config.num_classes,
            depth=config.depth,
            sm_dim=config.sm_dim,
            sm_patch_size=config.sm_patch_size,
            sm_enc_depth=config.sm_enc_depth,
            sm_enc_heads=config.sm_enc_heads,
            sm_enc_mlp_dim=config.sm_enc_mlp_dim,
            lg_dim=config.lg_dim,
            lg_patch_size=config.lg_patch_size,
            lg_enc_depth=config.lg_enc_depth,
            lg_enc_heads=config.lg_enc_heads,
            lg_enc_mlp_dim=config.lg_enc_mlp_dim,
            cross_attn_depth=config.cross_attn_depth,
            cross_attn_heads=config.cross_attn_heads,
            dropout=config.dropout,
            emb_dropout=config.emb_dropout
        )
        print("CrossViT model initialized successfully!")
    except Exception as e:
        print(f"Error initializing CrossViT model: {str(e)}")
        raise # Re-raise the exception to indicate failure

    # Create a sample input image
    img = torch.randn(1, 3, config.image_size, config.image_size)
    
    # Attempt to perform a forward pass
    try:
        print("Attempting forward pass...")
        preds = model(img)
        print(f"Forward pass successful. Output shape: {preds.shape}")
    except Exception as e:
        print(f"Error during forward pass: {str(e)}")
        raise # Re-raise the exception to indicate failure

if __name__ == "__main__":
    config = CrossViTConfig()
    try:
        initialize_and_test_crossvit(config)
    except Exception as e:
        print(f"Bug reproduction failed: {e}")
    else:
        print("Bug reproduction completed without errors (if no error was expected).")
