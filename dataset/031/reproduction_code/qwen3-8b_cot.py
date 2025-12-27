import torch
import torchvision.models as models

# Step 1: Define a standard ResNet50 model
model = models.resnet50(pretrained=False)

# Step 2: Load the NVIDIA pretrained weights (assuming the file is in the current directory)
pretrained_weights = torch.load('nvidia_resnet50_200821.pth.tar')

# Step 3: Attempt to load the state_dict into the model
try:
    model.load_state_dict(pretrained_weights, strict=True)
except RuntimeError as e:
    print("Error loading state_dict:", e)