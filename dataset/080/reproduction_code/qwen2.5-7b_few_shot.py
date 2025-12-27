import torch
from torch.utils.data import TensorDataset, DataLoader
from vit_pytorch import ViT

def create_data_loader(data, labels, batch_size):
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size)

# Create a simple dataset with 100 samples
data = torch.randn(100, 224, 224, 3)  # 100 images
labels = torch.randint(0, 2, (100,))  # 2 classes

# Split into training and validation (validation is a subset of training)
train_data, val_data = data, data[:50]
train_labels, val_labels = labels, labels[:50]

train_loader = create_data_loader(train_data, train_labels, batch_size=32)
val_loader = create_data_loader(val_data, val_labels, batch_size=32)

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
    for images, labels in train_loader:
        outputs = model(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    val_correct = 0
    for images, labels in val_loader:
        outputs = model(images)
        val_correct += (outputs.argmax(dim=1) == labels).sum().item()
    
    val_acc = val_correct / len(val_labels)
    print(f"Epoch {epoch} - Validation Accuracy: {val_acc:.4f}")  # Training loss is not calculated in this loop