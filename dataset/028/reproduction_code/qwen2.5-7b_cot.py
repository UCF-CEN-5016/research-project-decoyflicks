import torch
from torchvision import datasets, transforms
from vision_transformer import VisionTransformer  # Assuming the model is defined here

def load_cifar10_dataset(transform):
    return datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

def print_default_num_classes(model):
    print(f"Default num_classes: {model.num_classes}")

if __name__ == "__main__":
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR10 dataset
    train_dataset = load_cifar10_dataset(transform)

    # Initialize VisionTransformer with default num_classes
    model = VisionTransformer()

    # Check the default num_classes value
    print_default_num_classes(model)