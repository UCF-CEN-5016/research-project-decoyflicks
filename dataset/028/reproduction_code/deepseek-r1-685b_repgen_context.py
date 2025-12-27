import torch
from torchvision import datasets, transforms
from torchvision.models.vision_transformer import VisionTransformer

def get_cifar10_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_data = datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True,
        transform=transform
    )
    
    return train_data

def create_vision_transformer_model(num_classes=10):
    model = VisionTransformer(
        image_size=32,  # CIFAR10 image size
        patch_size=4,   # Default from example
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        num_classes=num_classes
    )
    
    return model

# Load CIFAR10 dataset
train_data = get_cifar10_dataset()

# Create model with default parameters and fix the bug
model = create_vision_transformer_model(num_classes=len(train_data.classes))

# Verify the fix
print(f"Model num_classes: {model.heads.head.out_features}")  # Should be 10
print(f"Actual dataset classes: {len(train_data.classes)}")