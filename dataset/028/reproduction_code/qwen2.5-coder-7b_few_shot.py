import torchvision
import torchvision.transforms as transforms
from torchvision.models import vit_b_16


def get_cifar10_transform():
    """Create the transform pipeline used for CIFAR-10 preprocessing."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def load_vit_model(num_classes: int = 10):
    """Instantiate a ViT-B/16 model configured for the given number of classes."""
    return vit_b_16(num_classes=num_classes)


def load_cifar10_dataset(root: str = './data', train: bool = True, download: bool = True, transform=None):
    """Load the CIFAR-10 dataset with the provided transform."""
    return torchvision.datasets.CIFAR10(root=root, train=train, download=download, transform=transform)


def main():
    transform_pipeline = get_cifar10_transform()
    model = load_vit_model(num_classes=10)
    cifar10_train = load_cifar10_dataset(root='./data', train=True, download=True, transform=transform_pipeline)

    # Check the output dimension of the model
    print(f"Model output dimension: {model.num_classes}")


if __name__ == '__main__':
    main()