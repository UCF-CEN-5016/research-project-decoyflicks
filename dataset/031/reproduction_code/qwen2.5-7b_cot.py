import torch
import torchvision.models as models

def load_pretrained_resnet50(model, weights_path):
    model.load_state_dict(torch.load(weights_path))
    return model

# Step 1: Define a standard ResNet50 model
model = models.resnet50(pretrained=False)

# Step 2: Load the NVIDIA pretrained weights (assuming the file is in the current directory)
weights_path = 'nvidia_resnet50_200821.pth.tar'

# Step 3: Load the pretrained weights into the model
try:
    model = load_pretrained_resnet50(model, weights_path)
except RuntimeError as e:
    print("Error loading state_dict:", e)