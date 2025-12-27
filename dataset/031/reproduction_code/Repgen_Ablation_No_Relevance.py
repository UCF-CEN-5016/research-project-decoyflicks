import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from timm.models.helpers import load_checkpoint

# Define transformation for the dataset
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the dataset
dataset = torchvision.datasets.ImageFolder(root='/data/imagenet/', transform=transform)
dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

# Load the pretrained model
model = models.resnet50(pretrained=True)

# Set the model to evaluation mode
model.eval()

# Load the state_dict from the file
state_dict = torch.load('nvidia_resnet50_200821.pth.tar')

# Attempt to load the state_dict into the model
try:
    model.load_state_dict(state_dict, strict=True)
except Exception as e:
    print(f"Error loading state_dict: {e}")