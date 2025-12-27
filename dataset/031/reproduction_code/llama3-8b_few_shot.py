import torch
from torchvision import models

# Load pretrained ResNet50 model
resnet50 = models.resnet50(pretrained=True)

# Attempt to load state dict from file
state_dict = torch.load('nvidia_resnet50_200821.pth.tar', map_location=torch.device('cpu'))
resnet50.load_state_dict(state_dict)

print("Loaded state dict")