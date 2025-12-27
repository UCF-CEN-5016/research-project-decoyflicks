import torch
from torchvision import datasets, transforms
from vision_transformer import VisionTransformer

# Minimal environment setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# Triggering conditions: Using vision_transformer with CIFAR10 dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Reproduce the bug: Use vision_transformer with default num_classes (16) for CIFAR10 dataset
model = VisionTransformer(num_classes=16)

print("Expected num_classes:", 10)
print("Actual num_classes:", model.num_classes)