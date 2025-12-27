import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ViTForImageClassification, ViTConfig
import numpy as np

# Ensure deterministic behavior
torch.manual_seed(42)
np.random.seed(42)

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Prepare synthetic dataset
def create_synthetic_dataset():
    # Generate 100 examples for demonstration
    X = np.random.randn(100, 224, 224, 3)  # Random images
    y = np.random.randint(0, 2, (100,))
    return SimpleDataset(X, y)

# Create dataloaders with small batch size to ensure diverse training batches
train_batch_size = 8
val_batch_size = 32

def create_dataloaders(dataset):
    train_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader

# Initialize model and training setup
model = ViTForImageClassification(
    num_classes=2,
    img_size=224,
    patch_size=32,
    embed_dim=128,
    depth=12,
    num_heads=8,
    mlp_ratio=4,
    qkv_bias=True,
    dropout=0.1
).to('cuda')

# Training loop setup (simplified for illustration)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, step_size_up=1000)

def train Epoch(model, train_loader, optimizer):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

# Validation setup
def validateEpoch(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    for inputs, labels in val_loader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Simulate initial training
for epoch in range(5):
    print(f'Epoch {epoch + 1}')
    train Epoch(model, train_loader, optimizer)
    val_acc = validateEpoch(model, val_loader)
    print(f'Validation Accuracy: {val_acc:.4f}')

    # Early exit if validation accuracy hits 100%
    if val_acc >= 1.0:
        break