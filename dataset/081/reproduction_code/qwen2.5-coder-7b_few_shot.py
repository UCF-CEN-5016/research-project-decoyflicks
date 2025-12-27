import torch
from vit_pytorch.cross_vit import CrossViT
from typing import Any


def build_crossvit_model(image_size: int, num_classes: int) -> Any:
    """
    Construct a CrossViT model with fixed architecture hyperparameters.
    The model expects 3-channel input images.
    """
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
        cross_attn_heads=8,
        dropout=0.1,
        emb_dropout=0.1
    )
    return model


def generate_single_channel_image(image_size: int) -> torch.Tensor:
    """
    Generate a single-channel random image tensor with shape (1, 1, H, W).
    """
    return torch.randn(1, 1, image_size, image_size)


if __name__ == "__main__":
    # Initialize CrossViT model (configured for 3-channel images)
    crossvit = build_crossvit_model(image_size=256, num_classes=1000)

    # Create a single-channel image tensor (shape: batch=1, channels=1, H, W)
    single_channel_img = generate_single_channel_image(image_size=256)

    # Attempt to run the model on the single-channel input (will raise an error)
    prediction = crossvit(single_channel_img)  # expected shape (1, 1000) if channels matched