import torch
from pytorch_examples.vision_transformer import VisionTransformer

def print_num_classes(model):
    print(f"Number of classes: {model.num_classes}")

# Initialize model with incorrect num_classes=16 for CIFAR10 (should be 10)
model = VisionTransformer(num_classes=16)

print_num_classes(model)