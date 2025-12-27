import torch
from vit_pytorch.cross_vit import CrossViT

def create_cross_vit_model():
    model = CrossViT(
        image_size=256,
        num_classes=10,
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
    return model

def main():
    # Create model
    model = create_cross_vit_model()

    # Input tensor with correct channel dimensions
    x = torch.randn(1, 3, 256, 256)

    # Model prediction
    out = model(x)
    print(out.shape)

if __name__ == "__main__":
    main()