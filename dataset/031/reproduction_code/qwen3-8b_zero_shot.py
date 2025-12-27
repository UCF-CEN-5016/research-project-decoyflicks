import torch
import torch.nn as nn
import torchvision.models as models

# Define a custom model that expects keys under 'layers.0.0.conv1.weight', etc.
class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        ])

    def forward(self, x):
        return self.layers[0](x)

# Load a standard ResNet50 model and get its state_dict
standard_model = models.resnet50(pretrained=True)
standard_state_dict = standard_model.state_dict()

# Create a custom model
custom_model = CustomResNet()

# Attempt to load the standard state_dict into the custom model
# This will raise a KeyError due to mismatched keys
custom_model.load_state_dict(standard_state_dict)

# Example: Rename keys to match the model's expected structure
renamed_state_dict = {}
for key, value in standard_state_dict.items():
    if key.startswith('layer1.0'):
        new_key = key.replace('layer1.0', 'layers.0.0')
        renamed_state_dict[new_key] = value

# Now load the renamed state_dict into the custom model
custom_model.load_state_dict(renamed_state_dict)