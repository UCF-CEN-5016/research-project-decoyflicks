import torch
from torchvision.datasets import CIFAR10
from vision_transformer import VisionTransformer

num_classes = 16  # Expected 10 for CIFAR10 dataset

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = CIFAR10(root='.', train=True, transform=transform)
test_dataset = CIFAR10(root='.', train=False, transform=transform)

vision_transformer_model = VisionTransformer(num_classes=num_classes)