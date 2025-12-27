import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import VisionTransformer

# Set up the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the hyperparameters
batch_size = 32
num_classes = 16  # Incorrect number of classes for CIFAR10
image_size = 224
patch_size = 16

# Load the CIFAR10 dataset
transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# Initialize the Vision Transformer model with the incorrect number of classes
model = VisionTransformer(img_size=image_size, patch_size=patch_size, num_classes=num_classes, embed_dim=768)

# Move the model to the device (GPU if available)
model.to(device)

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop (for demonstration purposes, we'll just do one epoch)
for epoch in range(1):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, loss = {running_loss / (i+1)}')

print("Model's number of classes:", model.num_classes)