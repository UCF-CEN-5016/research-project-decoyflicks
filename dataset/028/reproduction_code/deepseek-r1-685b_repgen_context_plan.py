import torch
from torchvision import datasets, transforms
from torchvision.models.vision_transformer import VisionTransformer

# Define transformations for the dataset
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
    image_size=32,
    patch_size=4,
    num_layers=12,
    num_heads=12,
    hidden_dim=768,
    mlp_dim=3072,
    num_classes=10  # Set num_classes to 10 for CIFAR10 dataset
)

# Verify the corrected num_classes
print(f"Model num_classes: {model.head.out_features}")  # Should be 10
print(f"Actual dataset classes: {len(train_data.classes)}")  # Should be 10