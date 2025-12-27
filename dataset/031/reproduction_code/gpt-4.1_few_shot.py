import torch
import torchvision.models as models

# Instantiate standard torchvision ResNet50 model
model = models.resnet50()

# Simulate loading a checkpoint with mismatched keys
# For example, checkpoint keys use 'layer1', but model expects 'layers'
checkpoint_state_dict = {
    'layer1.0.conv1.weight': torch.randn(64, 64, 1, 1),
    'layer1.0.bn1.weight': torch.randn(64),
    'layer1.0.bn1.bias': torch.randn(64),
    # ... (other keys omitted for brevity)
}

try:
    # This will raise RuntimeError due to unexpected/missing keys
    model.load_state_dict(checkpoint_state_dict)
except RuntimeError as e:
    print("RuntimeError caught during load_state_dict:")
    print(e)