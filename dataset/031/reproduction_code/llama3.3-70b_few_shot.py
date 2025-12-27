import torch
import torch.nn as nn
import torchvision.models as models

# Load a ResNet50 model
model = models.resnet50()

# Load the pretrained weights
pretrained_weights = torch.load('nvidia_resnet50_200821.pth.tar', map_location=torch.device('cpu'))

# Attempt to load the pretrained weights into the model
model.load_state_dict(pretrained_weights['state_dict'])

# This will cause an error due to mismatched key names
print("Model weights:", model.state_dict().keys())