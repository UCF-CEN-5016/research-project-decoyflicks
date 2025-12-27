import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import vision_transformer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a dummy dataset
num_samples = 100
image_size = 224
input_dim = 3 * image_size * image_size

# Generate dummy images
images = torch.rand(num_samples, 3, image_size, image_size)
labels = torch.zeros(num_samples, dtype=torch.long)

# Create DataLoader
dataset = TensorDataset(images, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define ViT model
model = vision_transformer.vit_base_patch16_224(pretrained=False)
model.head = nn.Linear(model.head.in_features, 2)  # Adjust output to 2 classes
model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Epoch {epoch+1} - loss: {total_loss/len(dataloader):.4f} - acc: {accuracy:.4f}")