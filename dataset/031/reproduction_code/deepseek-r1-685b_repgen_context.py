import torch
import torchvision.models as models

# Create model and load pretrained weights
model = models.resnet50(pretrained=False)  # Start with fresh model

# Simulate loading incompatible weights (keys don't match model architecture)
pretrained_weights = {
    "conv1.weight": torch.randn(64, 64, 3, 3),
    "bn1.weight": torch.randn(64),
    # ... other mismatched keys
}

# Attempt to load - will raise RuntimeError
try:
    model_dict = model.state_dict()
    pretrained_weights = {k: v for k, v in pretrained_weights.items() if k in model_dict}
    model_dict.update(pretrained_weights)
    model.load_state_dict(model_dict)
except RuntimeError as e:
    print("Error loading state_dict:")
    print(e)

# Show expected vs actual keys
print("\nExpected first few keys:")
print(list(model.state_dict().keys())[:5])  # Shows 'conv1.weight', 'bn1.weight' etc.
print("\nAttempted first few keys:")
print(list(pretrained_weights.keys())[:5])  # Shows 'conv1.weight', 'bn1.weight' etc.