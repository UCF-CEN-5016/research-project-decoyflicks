import torch
import torchvision.models as models
from typing import Dict, Any


def apply_pretrained_weights(model: torch.nn.Module, pretrained_weights: Dict[str, Any]) -> None:
    """
    Update the model's state dict with weights from a pretrained dictionary for matching keys only.
    This performs a non-strict update: only keys present in both the model and the pretrained dict are applied.
    """
    current_state = model.state_dict()
    compatible_weights = {k: v for k, v in pretrained_weights.items() if k in current_state}
    current_state.update(compatible_weights)
    model.load_state_dict(current_state)


def main() -> None:
    # Simulated pretrained weights
    pretrained_weights = {
        'layer1.0.conv1.weight': torch.randn(64, 3, 7, 7),
        'layer1.0.bn1.weight': torch.randn(64),
        'layer1.0.bn1.bias': torch.randn(64),
        # ... other keys from the original ResNet50 ...
        # But now, we introduce a mismatch
        'layer2.0.conv1.weight': torch.randn(128, 64, 1, 1),
        'layers.0.conv1.weight': torch.randn(64, 3, 7, 7),  # Mismatched key
    }

    # Define ResNet50 model
    model = models.resnet50(pretrained=False)

    # Load pretrained weights without strict checking
    try:
        apply_pretrained_weights(model, pretrained_weights)
    except RuntimeError as e:
        print("Error loading state_dict:", e)


if __name__ == "__main__":
    main()