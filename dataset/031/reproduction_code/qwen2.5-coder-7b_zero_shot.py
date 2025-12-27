import torch
from torchvision import models
from typing import Dict, Any


def create_pretrained_resnet50() -> torch.nn.Module:
    """Create a ResNet-50 model initialized with pretrained weights."""
    return models.resnet50(pretrained=True)


def load_state_dict_from_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load and return the 'state_dict' entry from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path)
    return checkpoint['state_dict']


def main() -> None:
    """Load a pretrained ResNet-50 and apply a checkpoint state_dict to it."""
    resnet_model = create_pretrained_resnet50()
    checkpoint_state = load_state_dict_from_checkpoint('nvidia_resnet50_200821.pth.tar')
    resnet_model.load_state_dict(checkpoint_state)


if __name__ == '__main__':
    main()