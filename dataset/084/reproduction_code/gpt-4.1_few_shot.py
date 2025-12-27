import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

# Dummy dataset (classification with 2 classes)
X = torch.randn(100, 3, 224, 224)
y = torch.randint(0, 2, (100,))

# Incorrect dataset split: train and val use the exact same data
train_dataset = TensorDataset(X, y)
val_dataset = TensorDataset(X, y)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

# Simple model to simulate ViT-like behavior (placeholder)
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(3 * 224 * 224, 2)

    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(3):
    model.train()
    total_correct = 0
    total_samples = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        total_correct += (preds == yb).sum().item()
        total_samples += yb.size(0)

    train_acc = total_correct / total_samples

    # Validation on the exact same data
    model.eval()
    val_correct = 0
    val_samples = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == yb).sum().item()
            val_samples += yb.size(0)

    val_acc = val_correct / val_samples

    print(f"Epoch: {epoch+1} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")