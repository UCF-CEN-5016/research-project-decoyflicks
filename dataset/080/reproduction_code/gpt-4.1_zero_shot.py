import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from vit_pytorch.vit_for_small_dataset import ViT

torch.manual_seed(0)

batch_size = 64

x_train = torch.randn(512, 3, 224, 224)
y_train = torch.randint(0, 2, (512,))
x_val = torch.randn(128, 3, 224, 224)
y_val = torch.randint(0, 2, (128,))

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size)

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    train_correct = 0
    train_total = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        opt.step()
        preds = out.argmax(dim=1)
        train_correct += (preds == yb).sum().item()
        train_total += yb.size(0)

    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds = out.argmax(dim=1)
            val_correct += (preds == yb).sum().item()
            val_total += yb.size(0)

    print(f'Epoch {epoch+1} Train Acc: {train_correct/train_total:.4f} Val Acc: {val_correct/val_total:.4f}')