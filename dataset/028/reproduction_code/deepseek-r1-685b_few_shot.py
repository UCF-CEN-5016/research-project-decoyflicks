import torch
from torchvision import datasets, transforms
from torchvision.models.vision_transformer import VisionTransformer

# Minimal setup to reproduce the bug
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR10 dataset
train_data = datasets.CIFAR10(
    root='./data', 
    train=True,
    download=True,
    transform=transform
)

# Create model with default parameters
model = VisionTransformer(
    image_size=32,  # CIFAR10 image size
    patch_size=4,   # Default from example
    num_layers=12,
    num_heads=12,
    hidden_dim=768,
    mlp_dim=3072,
    # The bug: num_classes defaults to 16 instead of 10
)

# Verify the mismatch
print(f"Model num_classes: {model.heads.head.out_features}")  # Should be 10 but shows 16
print(f"Actual dataset classes: {len(train_data.classes)}")   # Shows 10