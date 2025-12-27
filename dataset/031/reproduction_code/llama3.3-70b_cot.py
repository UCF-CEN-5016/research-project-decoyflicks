import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Minimal setup: Define a ResNet50 model (using torchvision for simplicity)
model = torchvision.models.resnet50(pretrained=False)

# Load the pretrained weights
# This step assumes the weights file is in the same directory
weights_file = 'nvidia_resnet50_200821.pth.tar'
state_dict = torch.load(weights_file, map_location='cpu')

# Attempt to load the state_dict into the model
# This will trigger the bug due to the mismatch in keys
try:
    model.load_state_dict(state_dict['state_dict'], strict=True)
except RuntimeError as e:
    print(f"Error loading state_dict: {e}")

# To fix this, one might need to adjust the model architecture to match the keys in the weights file
# or use a less strict loading method, though this requires caution to ensure the model is correctly initialized