import torch
from torchvision import datasets, transforms
from vision_transformer import VisionTransformer  # Assuming the model is defined here

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR10 dataset (not used in this reproduction, but required for context)
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Initialize VisionTransformer with default num_classes
model = VisionTransformer()

# Check the default num_classes value
print(f"Default num_classes: {model.num_classes}")