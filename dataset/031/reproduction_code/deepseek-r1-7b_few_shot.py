import torch
from torchvision import models

# Load the pretrained ResNet50v1.5 model which might have different key naming convention
model = models.ResNet50_v1_5()

# Path to the downloaded pretrained weights (adjust this path as needed)
weights_path = "nvidia_resnet50_200821.pth.tar"

try:
    # Attempt to load state_dict with custom renaming of keys to match expected structure
    state_dict = torch.load(weights_path, 
                           map_location=lambda storage, loc: storage.rename(
                               r'.layers.0.0 Conv1: $ storage[\'layer1.0.conv1.weight\']',  # Renaming conv1 layer 0
                               r'.layers.0.1 Conv2: $ storage[\'layer1.0.conv2.weight\']'))  # Renaming conv2 layer 0

    model.load_state_dict(state_dict)
    
except RuntimeError as e:
    print(f"Error loading state_dict: {e}")