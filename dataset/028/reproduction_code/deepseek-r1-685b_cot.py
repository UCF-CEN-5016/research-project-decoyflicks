import torch
from torchvision import datasets
from vision_transformer import VisionTransformer

# Setup minimal environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CIFAR10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True)

# Initialize model with default parameters
model = VisionTransformer().to(device)

# Check the output layer size
print(f"Model output classes: {model.head.out_features}")  # Will show 16 when it should be 10

# Correct usage - explicitly specify num_classes
correct_model = VisionTransformer(num_classes=10).to(device)
print(f"Correct model output classes: {correct_model.head.out_features}")  # Shows 10