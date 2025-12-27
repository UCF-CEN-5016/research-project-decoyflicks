import torch
from vision_transformer import ViT, VITParams

# Ensure that num_classes is set explicitly to 10 for CIFAR-10 dataset
model = ViT(num_classes=10)
print(f"Number of classes in model: {model.num_classes}")