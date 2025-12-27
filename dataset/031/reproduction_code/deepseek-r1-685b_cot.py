import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck

# Simplified ResNet model definition with different layer naming
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Sequential(  # This creates the naming mismatch
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        )

# Create model and attempt to load pretrained weights
model = ResNet()
try:
    # Simulate loading weights that use standard torchvision naming
    pretrained_weights = {
        'layer1.0.conv1.weight': torch.rand(64, 64, 1, 1),
        'layer1.0.bn1.weight': torch.rand(64),
        'layer1.0.bn1.bias': torch.rand(64),
    }
    model.load_state_dict(pretrained_weights, strict=True)
except RuntimeError as e:
    print("Reproduced error:")
    print(e)