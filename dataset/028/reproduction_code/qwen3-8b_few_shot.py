import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vit_b_16

# Incorrect default num_classes (16) for CIFAR10 (should be 10)
model = vit_b_16(num_classes=16)  # This would be the default in buggy code

# CIFAR10 dataset (10 classes)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Model output shape mismatch
print(f"Model output dimension: {model.classifier[2].out_features}")  # Should be 10, but is 16