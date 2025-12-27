import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torch import nn, optim
from efficientvit.models import ViT, Linformer

# Create dummy dataset (all samples are the same)
data = torch.randn(32, 3, 224, 224)  # Dummy images
labels = torch.zeros(32, 1).squeeze()  # All samples labeled as class 0

# Create identical training and validation datasets
train_dataset = TensorDataset(data, labels)
val_dataset = TensorDataset(data, labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model setup (same as in the original example)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
efficient_transformer = Linformer(
    dim=128,
    seq_len=49 + 1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)
model = ViT(
    dim=128,
    image_size=224,
    patch_size=32,
    num_classes=2,
    transformer=efficient_transformer,
    channels=3,
).to(device)

# Dummy optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training loop that produces 100% accuracy
for epoch in range(5):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        print(f"Epoch {epoch+1} - loss: {val_loss/len(val_loader):.4f} - val_acc: {correct/total:.4f}")