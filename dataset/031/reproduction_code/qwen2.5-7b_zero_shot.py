import torch
import torch.nn as nn
import torchvision.models as models

# Define a custom model that expects keys under 'layers.0.0.conv1.weight', etc.
class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

# Load a standard ResNet50 model and get its state_dict
standard_model = models.resnet50(pretrained=True)
standard_state_dict = standard_model.state_dict()

# Create a custom model
custom_model = CustomResNet()

# Map the keys of the standard state_dict to match the custom model's structure
mapped_state_dict = {}
for key, value in standard_state_dict.items():
    if key.startswith('layer1.0'):
        new_key = key.replace('layer1.0', 'conv1')
        mapped_state_dict[new_key] = value

# Load the mapped state_dict into the custom model
custom_model.load_state_dict(mapped_state_dict)