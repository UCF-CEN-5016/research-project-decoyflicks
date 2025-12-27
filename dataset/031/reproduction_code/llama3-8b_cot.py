import torch
import torchvision.models as models

# Set up minimal environment
torch.backends.cudnn.benchmark = False  # For CPU-only testing

# Load pretrained ResNet50 model and weights
resnet50_model = models.resnet50(pretrained=True)

# Print the state_dict to see the expected keys
print(resnet50_model.state_dict().keys())

# Wrap final code in a try-except block to catch any errors
try:
    # Load the state_dict with the missing/expected keys
    state_dict = torch.load('nvidia_resnet50_200821.pth.tar')['state_dict']
    resnet50_model.load_state_dict(state_dict)

    print("Loaded state_dict successfully!")
except RuntimeError as e:
    print(f"Error loading state_dict: {e}")