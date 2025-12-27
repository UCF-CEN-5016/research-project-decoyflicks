import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vit_b_16

# Load ViT-B/16 model with correct num_classes for CIFAR-10
model = vit_b_16(num_classes=10)

# CIFAR-10 dataset (10 classes)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Check the output dimension of the model
print(f"Model output dimension: {model.num_classes}")