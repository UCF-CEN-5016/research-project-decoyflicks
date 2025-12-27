import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
from vit_model import ViT  # Assuming ViT model is defined in vit_model.py
import device_setting  # Assuming device setting is done in device_setting.py

# Set up CIFAR10 dataset with transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_data = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
valid_data = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)

# Create DataLoader objects
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=4, shuffle=True)

# Initialize ViT model with default arguments
args = argparse.Namespace(patch_size=16, latent_size=768, n_channels=3, num_heads=12, num_encoders=12, dropout=0.1, img_size=224, num_classes=10, epochs=10, lr=1e-2, weight_decay=3e-2)
model = ViT(args).to(device)

# Create an instance of TrainEval and call the train method
train_eval = TrainEval(args, model, train_loader, valid_loader, optimizer, criterion, device)
train_eval.train()

# Load best weights
best_weights_path = 'best-weights.pt'
model.load_state_dict(torch.load(best_weights_path))

# Verify num_classes parameter
print(f"Number of classes in ViT model: {model.num_classes}")

# Create a new CIFAR10 test dataset and DataLoader
test_data = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=4, shuffle=False)

# Evaluate the model on the test dataset
validation_loss = train_eval.eval_fn(0)  # Assuming eval_fn takes an epoch index as argument

# Assert validation loss
assert validation_loss > 2.30, "Validation loss is not greater than 2.30"