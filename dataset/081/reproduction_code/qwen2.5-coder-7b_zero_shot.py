import torch
from typing import Dict
from vit_pytorch.cross_vit import CrossViT


def build_crossvit_model(config: Dict) -> CrossViT:
    return CrossViT(
        image_size=config["image_size"],
        num_classes=config["num_classes"],
        depth=config["depth"],
        sm_dim=config["sm_dim"],
        sm_patch_size=config["sm_patch_size"],
        sm_enc_depth=config["sm_enc_depth"],
        sm_enc_heads=config["sm_enc_heads"],
        sm_enc_mlp_dim=config["sm_enc_mlp_dim"],
        lg_dim=config["lg_dim"],
        lg_patch_size=config["lg_patch_size"],
        lg_enc_depth=config["lg_enc_depth"],
        lg_enc_heads=config["lg_enc_heads"],
        lg_enc_mlp_dim=config["lg_enc_mlp_dim"],
        cross_attn_depth=config["cross_attn_depth"],
        cross_attn_heads=config["cross_attn_heads"],
        dropout=config["dropout"],
        emb_dropout=config["emb_dropout"],
    )


def create_dummy_image(batch: int, channels: int, height: int, width: int) -> torch.Tensor:
    return torch.randn(batch, channels, height, width)


def main() -> None:
    config = {
        "image_size": 256,
        "num_classes": 1000,
        "depth": 4,
        "sm_dim": 192,
        "sm_patch_size": 16,
        "sm_enc_depth": 2,
        "sm_enc_heads": 8,
        "sm_enc_mlp_dim": 2048,
        "lg_dim": 384,
        "lg_patch_size": 64,
        "lg_enc_depth": 3,
        "lg_enc_heads": 8,
        "lg_enc_mlp_dim": 2048,
        "cross_attn_depth": 2,
        "cross_attn_heads": 8,
        "dropout": 0.1,
        "emb_dropout": 0.1,
    }

    model = build_crossvit_model(config)
    input_tensor = create_dummy_image(batch=1, channels=3, height=256, width=256)
    pred = model(input_tensor)  # (1, 1000)
    return pred


if __name__ == "__main__":
    _ = main()