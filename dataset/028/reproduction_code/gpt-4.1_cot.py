import torch
import torchvision
import torchvision.transforms as transforms
from vision_transformer import VisionTransformer  # assuming vision_transformer.py or module is accessible

def reproduce_bug():
    # Minimal transform for CIFAR10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load CIFAR10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    # Here is the key place: instantiate VisionTransformer with default num_classes
    # The bug is that num_classes defaults to 16 instead of 10
    model = VisionTransformer()

    print(f"Model num_classes attribute: {model.num_classes}")
    # Expected: 10, Actual (bug): 16

if __name__ == "__main__":
    reproduce_bug()