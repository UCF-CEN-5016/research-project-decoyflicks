import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from efficient_transformer import Linformer, ViT

# Step 1: Create a dummy dataset with identical images and labels
def create_dummy_dataset(num_samples=100, image_channels=3, image_size=224):
    images = torch.rand(num_samples, image_channels, image_size, image_size)
    labels = torch.zeros(num_samples, dtype=torch.long)
    return TensorDataset(images, labels)

train_dataset = create_dummy_dataset()
val_dataset = create_dummy_dataset()

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Step 2: Define the ViT model with Linformer
model = ViT(
    image_size=224,
    patch_size=32,
    num_classes=2,
    dim=64,
    depth=6,
    heads=8,
    mlp_dim=128,
    dropout=0.1,
    emb_dropout=0.1
)

# Step 3: Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate
        val_accuracy = evaluate_model(model, val_loader)
        print(f"Epoch {epoch + 1}, Validation Accuracy: {val_accuracy:.2f}%")

def evaluate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Train the model
train_model(model, train_loader, criterion, optimizer)