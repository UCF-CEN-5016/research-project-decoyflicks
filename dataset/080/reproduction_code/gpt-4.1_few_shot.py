import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from vit_pytorch.vit_for_small_dataset import ViT

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64

# Dummy dataset (e.g. cats and dogs binary classification)
X_train = torch.randn(500, 3, 224, 224)
y_train = torch.randint(0, 2, (500,))
X_val = torch.randn(100, 3, 224, 224)
y_val = torch.randint(0, 2, (100,))

train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=batch_size)
val_loader = DataLoader(val_ds, batch_size=batch_size)

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
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training and validation loop with a bug:
# Validation done WITHOUT model.eval() causing dropout active in validation,
# leading to unstable/incorrect validation behavior (often higher val accuracy).

for epoch in range(5):
    model.train()
    train_correct = 0
    train_total = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        train_correct += (preds.argmax(1) == yb).sum().item()
        train_total += yb.size(0)

    train_acc = train_correct / train_total

    # BUG: Missing model.eval() here causes dropout active in validation
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)  # model is still in train() mode
            val_correct += (preds.argmax(1) == yb).sum().item()
            val_total += yb.size(0)
    val_acc = val_correct / val_total

    print(f"Epoch {epoch}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")