import torch
import torch.nn as nn
from torchvision.models import resnet50

# Step 1: Load a pretrained state_dict with keys like 'layer1.0.conv1.weight'
pretrained_model = resnet50(pretrained=True)
pretrained_state_dict = pretrained_model.state_dict()

# Step 2: Create a dummy model with different key names that expect 'layers' instead of 'layer'
class DummyResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        # We simulate the layers attribute name difference by renaming keys in original model
        self.model = resnet50(pretrained=False)

        # Rename layer1->layers. This simulates the repo's model
        # But since model parameters are fixed, we replace the state_dict keys later

    def forward(self, x):
        return self.model(x)

    def state_dict(self, *args, **kwargs):
        # Override to rename keys from 'layer1' to 'layers' to simulate model expectation
        orig = self.model.state_dict(*args, **kwargs)
        renamed = {}
        for k, v in orig.items():
            # Rename 'layer1' -> 'layers.0', 'layer2' -> 'layers.1', etc. (example)
            if k.startswith('layer1'):
                new_key = k.replace('layer1', 'layers.0')
            elif k.startswith('layer2'):
                new_key = k.replace('layer2', 'layers.1')
            elif k.startswith('layer3'):
                new_key = k.replace('layer3', 'layers.2')
            elif k.startswith('layer4'):
                new_key = k.replace('layer4', 'layers.3')
            else:
                new_key = k
            renamed[new_key] = v
        return renamed

# Step 3: Instantiate dummy model
dummy_model = DummyResNet50()

# Step 4: Try to load pretrained weights directly (should fail due to key mismatch)
try:
    dummy_model.load_state_dict(pretrained_state_dict)
except RuntimeError as e:
    print("RuntimeError while loading state_dict:")
    print(e)