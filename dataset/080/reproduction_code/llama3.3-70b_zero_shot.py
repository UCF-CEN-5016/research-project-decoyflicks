import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from vit_pytorch import ViT

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters
batch_size = 64
image_size = 224
patch_size = 16
num_classes = 2
dim = 1024
depth = 6
heads = 16
mlp_dim = 2048
dropout = 0.1
emb_dropout = 0.1

# Load dataset
transform = transforms.Compose([transforms.Resize((image_size, image_size)), 
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = datasets.FakeData(size=1000, image_size=(3, image_size, image_size), num_classes=num_classes, transform=transform)
val_dataset = datasets.FakeData(size=100, image_size=(3, image_size, image_size), num_classes=num_classes, transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define model
model = ViT(image_size=image_size, 
            patch_size=patch_size, 
            num_classes=num_classes, 
            dim=dim, 
            depth=depth, 
            heads=heads, 
            mlp_dim=mlp_dim, 
            dropout=dropout, 
            emb_dropout=emb_dropout).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
for epoch in range(10):
    # Train
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_accuracy = correct / total
    train_loss /= len(train_loader)

    # Validate
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = correct / total
    val_loss /= len(val_loader)

    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')