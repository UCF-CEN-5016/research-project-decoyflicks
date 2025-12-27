import torch
from vit_pytorch.cross_vit import CrossViT

def test_crossvit_model(model, channels):
    try:
        img = torch.randn(1, channels, 256, 256)  # Batch of 1 image
        pred = model(img)
        return True, pred.shape
    except Exception as e:
        return False, str(e)

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
channel_tests = [1, 2, 4]
for channels in channel_tests:
    print(f"\nTesting with {channels} channel input:")
    success, output_shape = test_crossvit_model(model, channels)
    if success:
        print("Success! Output shape:", output_shape)
    else:
        print(f"Failed with {channels} channels. Error:", output_shape)