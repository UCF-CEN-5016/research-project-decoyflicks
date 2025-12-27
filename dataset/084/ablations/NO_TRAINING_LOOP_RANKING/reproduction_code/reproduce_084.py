import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from vit_pytorch.ats_vit import ViT

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dim = 128
image_size = 224
patch_size = 32
num_classes = 2
depth = 12
heads = 8
k = 64

data = torch.randn(1000, 3, image_size, image_size)
labels = torch.randint(0, num_classes, (1000,))

train_data = TensorDataset(data[:800], labels[:800])
val_data = TensorDataset(data[800:], labels[800:])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

model = ViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=dim, depth=depth, heads=heads, mlp_dim=dim*4).to(device)

model.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()

    train_accuracy = (outputs.argmax(dim=1) == lbls).float().mean().item()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}, Train Accuracy: {train_accuracy * 100:.2f}%')

    with torch.no_grad():
        val_loss = 0
        val_accuracy = 0
        for val_imgs, val_lbls in val_loader:
            val_imgs, val_lbls = val_imgs.to(device), val_lbls.to(device)
            val_outputs = model(val_imgs)
            val_loss += criterion(val_outputs, val_lbls).item()
            val_accuracy += (val_outputs.argmax(dim=1) == val_lbls).float().sum().item()

        val_loss /= len(val_loader)
        val_accuracy /= len(val_data)
        print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy * 100:.2f}%')

    assert train_accuracy > 0.99, "Training accuracy did not exceed 99%"
    if val_accuracy == 1.0:
        print("Bug reproduced: Validation accuracy reached 100% after the first epoch.")