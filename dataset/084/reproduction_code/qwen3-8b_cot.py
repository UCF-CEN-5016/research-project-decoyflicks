import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.models import vit_b_16

# 1. Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Create a dummy dataset with all labels = 0 (simulating a mislabeled dataset)
#    - 100 images of size 224x224 (3 channels)
#    - All labels are 0 (binary classification)
num_samples = 100
image_size = 224
input_dim = 3 * image_size * image_size  # 3 channels, 224x224

# Generate dummy images (random)
images = torch.rand(num_samples, 3, image_size, image_size)

# Labels: all 0
labels = torch.zeros(num_samples, dtype=torch.long)

# Create dataset and dataloader
dataset = TensorDataset(images, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. Define a simple ViT model (using torchvision's ViT for simplicity)
model = vit_b_16(pretrained=False).to(device)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)  # Adjust output to 2 classes
model = model.to(device)

# 4. Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 5. Training loop (simulated)
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
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Epoch {epoch+1} - loss: {total_loss/len(dataloader):.4f} - acc: {accuracy:.4f}")