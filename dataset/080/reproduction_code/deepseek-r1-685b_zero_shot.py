import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from vit_pytorch.vit_for_small_dataset import ViT
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# Create synthetic data
num_samples = 1000
x_train = torch.randn(num_samples, 3, 224, 224)
y_train = torch.randint(0, 2, (num_samples,))
x_val = torch.randn(num_samples//5, 3, 224, 224)
y_val = torch.randint(0, 2, (num_samples//5,))

train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# Model
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

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    train_correct = 0
    train_total = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        train_total += y.size(0)
        train_correct += (predicted == y).sum().item()
    
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for x, y in val_loader:
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            val_total += y.size(0)
            val_correct += (predicted == y).sum().item()
    
    print(f"Epoch {epoch+1}: Train Acc: {train_correct/train_total:.4f}, Val Acc: {val_correct/val_total:.4f}")