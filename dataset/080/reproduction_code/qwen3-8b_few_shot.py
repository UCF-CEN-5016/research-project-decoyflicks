import torch
from torch.utils.data import TensorDataset, DataLoader
from vit_pytorch import ViT

# Create a simple dataset with 100 samples
data = torch.randn(100, 224, 224, 3)  # 100 images
labels = torch.randint(0, 2, (100,))  # 2 classes

# Split into training and validation (validation is a subset of training)
train_data = data
train_labels = labels
val_data = data[:50]  # First 50 samples as validation
val_labels = labels[:50]

train_dataset = TensorDataset(train_data, train_labels)
val_dataset = TensorDataset(val_data, val_labels)

train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)

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

# Dummy training loop
for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        outputs = model(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        # Dummy optimizer step
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_correct = 0
        for images, labels in val_loader:
            outputs = model(images)
            val_correct += (outputs.argmax(dim=1) == labels).sum().item()
    
    train_loss = loss.item()
    val_acc = val_correct / len(val_labels)
    print(f"Epoch {epoch} - Training Loss: {train_loss:.4f}, Validation Accuracy: {val_acc:.4f}")