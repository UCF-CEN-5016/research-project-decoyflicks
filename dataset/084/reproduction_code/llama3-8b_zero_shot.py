import torch
from efficient_transformer import Linformer
from vit_pytorch import ViT
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

efficient_transformer = Linformer(
    dim=128,
    seq_len=49+1,  # 7x7 patches + 1 cls-token
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

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = dsets.CIFAR10(root='./data', train=True, download=True, transform=transform)
val_dataset = dsets.CIFAR10(root='./data', train=False, download=True, transform=transform)

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(6):
    model.train()
    total_loss = 0
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x.view(-1, 3, 224, 224))
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch : {epoch+1} - loss : {total_loss/(i+1)} - acc: ? - val_loss : ? - val_acc: ?')