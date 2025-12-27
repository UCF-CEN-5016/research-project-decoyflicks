import torch
from vit_pytorch.cross_vit import CrossViT

def test_cross_vit_model(model, channels):
    try:
        img = torch.randn(1, channels, 256, 256)  # Batch of 1 image
        pred = model(img)
        print("Success! Output shape:", pred.shape)
    except Exception as e:
        print(f"Failed with {channels} channels. Error:", str(e))

if __name__ == "__main__":
    # Initialize CrossViT model (default expects 3 channels)
    model = CrossViT(
        image_size=256,
        num_classes=1000,
        depth=4,
        sm_dim=192,
        sm_patch_size=16,
        lg_dim=384,
        lg_patch_size=64
    )

    # Try with different channel counts
    for channels in [1, 2, 4]:  # Fails on all non-3 channel inputs
        print(f"\nTesting with {channels} channel input:")
        test_cross_vit_model(model, channels)