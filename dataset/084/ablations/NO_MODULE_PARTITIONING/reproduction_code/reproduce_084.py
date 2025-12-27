import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from vit_pytorch import ViT  # Assuming the ViT class is defined in vit_pytorch module

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

dim = 128
image_size = 224
patch_size = 32
num_classes = 2
depth = 12
heads = 8
k = 64

# Create synthetic dataset
X = torch.randn(1000, 3, image_size, image_size)
y = torch.randint(0, num_classes, (1000,))

# Split dataset
train_dataset = TensorDataset(X[:800], y[:800])
val_dataset = TensorDataset(X[800:], y[800:])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the ViT model
model = ViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=dim, depth=depth, heads=heads, mlp_dim=dim*4, channels=3).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_accuracy = correct / total
    print(f'Epoch [{epoch+1}/5], Loss: {train_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.4f}')
    
    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_accuracy = correct / total
    print(f'Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}')
    
    assert train_accuracy > 0.99, "Training accuracy did not exceed 0.99"
    assert val_accuracy > 0.99, "Validation accuracy did not exceed 0.99"
    
    if train_accuracy == 1.0 and val_accuracy == 1.0:
        print("Bug reproduced: 100% accuracy achieved.")