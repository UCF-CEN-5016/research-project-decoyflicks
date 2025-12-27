import torch
from torch import nn
from vit_pytorch.na_vit_nested_tensor_3d import NaViT

# Minimal setup to reproduce the bug
def reproduce_bug():
    # Create nested tensor input (batch of variable-sized 3D images)
    images = [
        torch.randn(3, 64, 64),  # 3 channels, 64x64
        torch.randn(3, 128, 128),
        torch.randn(3, 96, 96),
        torch.randn(3, 32, 32),
        torch.randn(3, 64, 64)
    ]
    
    # Create model with same config that triggers the bug
    model = NaViT(
        image_size = 256,  # max size
        patch_size = 32,
        num_classes = 10,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048
    )
    
    # Forward pass
    preds = model(images)
    
    # Simulate training step that triggers backward pass
    loss = preds.sum()
    loss.backward()

# Run the reproduction
reproduce_bug()