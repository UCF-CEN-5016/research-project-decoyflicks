import torch
import torchvision.models as models

# Step 1: Load the NVIDIA-pretrained weights (simulated)
# In a real scenario, you'd load the weights from a file
# For demonstration, we'll simulate a mismatched state_dict
pretrained_dict = {
    'layer1.0.conv1.weight': torch.randn(64, 3, 7, 7),
    'layer1.0.bn1.weight': torch.randn(64),
    'layer1.0.bn1.bias': torch.randn(64),
    # ... other keys from the original ResNet50 ...
    # But now, we introduce a mismatch
    'layer2.0.conv1.weight': torch.randn(128, 64, 1, 1),
    'layers.0.conv1.weight': torch.randn(64, 3, 7, 7),  # Mismatched key
}

# Step 2: Define a standard ResNet50 model
model = models.resnet50(pretrained=False)

# Step 3: Attempt to load the state_dict, which will cause key mismatches
# The key 'layers.0.conv1.weight' is not present in the original model's state_dict
# and the key 'layer2.0.conv1.weight' might not match the actual model structure
try:
    model.load_state_dict(pretrained_dict, strict=False)
except RuntimeError as e:
    print("Error loading state_dict:", e)