import torch
from vit_pytorch.vit_for_small_dataset import ViT
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# Configuration
batch_size = 64
num_epochs = 5
learning_rate = 0.001
weight_decay = 1e-4

# Data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Create datasets
train_dataset = datasets.ImageFolder(root='.', transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(root='.', transform=data_transforms['val'])

# Create data loaders with proper shuffling and drop_last=True for consistency
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)

# Model definition
model = ViT(
    image_size=224,
    patch_size=16,
    num_classes=2,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)

# Loss and optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training loop
train_acc_history = []
val_acc_history = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        
        # Validation loop
        model.eval()
        val_images = next(iter(val_loader))
        val_outputs = model(val_images[0])
        val_loss = criterion(val_outputs.squeeze(), val_images[1].float())
        
        # Track metrics
        running_loss += loss.item()
        acc = (torch.round(torch.sigmoid(outputs)) == labels).sum().item() / len(outputs)
        val_acc = (torch.round(torch.sigmoid(val_outputs)) == val_images[1]).sum().item() / len(val_images[1])
        
    avg_train_loss = running_loss / len(train_loader)
    train_acc = np.mean([x['train_acc'] for x in history])  # Assuming 'history' is a list of dicts
    val_acc = np.mean([x['val_acc'] for x in history])
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss:.4f}')
    print(f'Training Accuracy: {train_acc:.4f}')
    print(f'Validation Accuracy: {val_acc:.4f}')