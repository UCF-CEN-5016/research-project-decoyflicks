import torch
from vit_pytorch.vit_for_small_dataset import ViT, SmallDatasetVit
from torch.utils.data import DataLoader

# Setup
batch_size = 64
patch_size = 16
image_size = 224
num_classes = 2
train_dir = "path/to/training"
val_dir = "path/to/validation"

# Create ViT model
model = ViT(
    image_size=image_size,
    patch_size=patch_size,
    num_classes=num_classes,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)

# Create dataset and data loaders
train_dataset = SmallDatasetVit(train_dir, image_size=image_size, patch_size=patch_size)
val_dataset = SmallDatasetVit(val_dir, image_size=image_size, patch_size=patch_size)

def create_dataloaders(train_dataset, val_dataset, batch_size):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return train_loader, val_loader

train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, batch_size)

# Training loop (simplified)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validation
    val_total = 0
    val_correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    train_acc = ... # Calculate training accuracy
    val_acc = val_correct / val_total if val_total else 0.0
    print(f"Epoch [{epoch+1}/10], Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}")