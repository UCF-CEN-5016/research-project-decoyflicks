import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from vit_pytorch import ViT
from vit_pytorch.efficient import Linformer

class SimpleDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.data = torch.randn(num_samples, 3, 224, 224)
        self.labels = torch.randint(0, 2, (num_samples,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_ds = SimpleDataset(1000)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_ds = SimpleDataset(200)
val_loader = DataLoader(val_ds, batch_size=32)

efficient_transformer = Linformer(
    dim=128,
    seq_len=49+1,
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

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(5):
    model.train()
    train_loss, train_correct = 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += (output.argmax(1) == y).sum().item()
    
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            val_loss += criterion(output, y).item()
            val_correct += (output.argmax(1) == y).sum().item()
    
    print(f'Epoch: {epoch+1} - loss: {train_loss/len(train_loader):.4f} - acc: {train_correct/len(train_ds):.4f} - val_loss: {val_loss/len(val_loader):.4f} - val_acc: {val_correct/len(val_ds):.4f}')