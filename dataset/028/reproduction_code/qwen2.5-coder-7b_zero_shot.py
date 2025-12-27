import torch
from typing import Tuple
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from vision_transformer import VisionTransformer

NUM_CLASSES = 16  # Expected 10 for CIFAR10 dataset
DATA_ROOT = "."


def build_transforms() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor()])


def load_cifar10_datasets(root: str, transform: transforms.Compose) -> Tuple[CIFAR10, CIFAR10]:
    train_ds = CIFAR10(root=root, train=True, transform=transform)
    test_ds = CIFAR10(root=root, train=False, transform=transform)
    return train_ds, test_ds


def create_vision_transformer(num_classes: int) -> VisionTransformer:
    return VisionTransformer(num_classes=num_classes)


# Prepare transforms and datasets
_default_transform = build_transforms()
train_dataset, test_dataset = load_cifar10_datasets(DATA_ROOT, _default_transform)

# Instantiate model
vision_transformer_model = create_vision_transformer(NUM_CLASSES)