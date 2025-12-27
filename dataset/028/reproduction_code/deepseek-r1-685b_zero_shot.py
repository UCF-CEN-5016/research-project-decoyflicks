import torch
import torchvision
from torchvision.models.vision_transformer import VisionTransformer

model = VisionTransformer(
    image_size=32,
    patch_size=4,
    num_layers=6,
    num_heads=8,
    hidden_dim=512,
    mlp_dim=2048
)

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
print(f"Expected num_classes: 10 (CIFAR10), Actual num_classes: {model.heads.head.out_features}")