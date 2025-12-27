import torch
import torchvision.models as models

def load_pretrained_weights(model, pretrained_dict):
    model_dict = model.state_dict()
    matched_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(matched_dict)
    model.load_state_dict(model_dict)

# Simulated pretrained weights
pretrained_dict = {
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
    load_pretrained_weights(model, pretrained_dict)
except RuntimeError as e:
    print("Error loading state_dict:", e)