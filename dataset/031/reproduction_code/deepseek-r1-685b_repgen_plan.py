import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os

# Dummy ResNet model to simulate the architecture mismatch
class ResNet(nn.Module):
    def __init__(self, version):
        super().__init__()
        self.version = version
        # These layer names are designed to produce 'missing keys' when
        # pretrained_weights with 'layerX.Y.convZ.weight' are loaded.
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64)
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128)
            )
        ])
        self.fc = nn.Linear(128, 1000)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.mean([2, 3])
        return self.fc(x)

# Dummy ImageNetDataset for demonstration purposes
class ImageNetDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self._data_len = 1000

    def __len__(self):
        return self._data_len

    def __getitem__(self, idx):
        return torch.randn(3, 224, 224)

def simulate_pretrained_weights(url: str) -> dict:
    """
    Simulates loading pretrained weights from a URL.
    The returned dictionary is structured to cause key mismatches
    with the `ResNet` class defined above.
    """
    print(f"Simulating loading of pretrained weights from: {url}")
    # These keys are designed to be 'unexpected keys' for our dummy ResNet
    # because our dummy ResNet has 'layers.0.0.weight' etc.
    dummy_state_dict = {
        'conv1.weight': torch.randn(64, 3, 7, 7),
        'bn1.weight': torch.randn(64),
        'bn1.bias': torch.randn(64),
        'layer1.0.conv1.weight': torch.randn(64, 64, 1, 1),
        'layer1.0.bn1.weight': torch.randn(64),
        'fc.weight': torch.randn(1000, 512),
        'fc.bias': torch.randn(1000)
    }
    print("Simulated pretrained weights dictionary created.")
    return dummy_state_dict

def create_model_instance(version: float) -> ResNet:
    """
    Creates an instance of the ResNet model and sets it to evaluation mode.
    """
    model = ResNet(version=version)
    model.eval()
    print(f"Model instance (ResNet version={version}) created and set to eval mode.")
    return model

def prepare_data_loader_instance(data_path: str, batch_size: int) -> DataLoader:
    """
    Prepares a DataLoader for a dummy ImageNet dataset.
    """
    dataset = ImageNetDataset(root_dir=data_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    print(f"DataLoader prepared for {len(dataset)} samples with batch size {batch_size}.")
    return data_loader

def attempt_and_verify_weight_loading(model: nn.Module, weights: dict):
    """
    Attempts to load weights into the model and asserts that the expected
    'missing keys' and 'unexpected keys' error occurs.
    """
    print("Attempting to load simulated pretrained weights into the model...")
    caught_exception = None
    try:
        model.load_state_dict(weights)
        raise RuntimeError("Expected state_dict loading error was not raised. Weights loaded successfully.")
    except Exception as e:
        caught_exception = e
        print(f"Caught expected exception during state_dict loading: {e}")

    if caught_exception:
        error_message = str(caught_exception)
        assert "missing keys" in error_message, f"Error message missing 'missing keys': {error_message}"
        assert "unexpected keys" in error_message, f"Error message missing 'unexpected keys': {error_message}"
        print("Successfully verified 'missing keys' and 'unexpected keys' in the error message.")
    else:
        raise AssertionError("No exception was caught, but one was expected for verification.")

