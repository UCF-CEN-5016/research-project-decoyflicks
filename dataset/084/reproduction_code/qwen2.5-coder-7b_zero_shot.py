import torch
from efficient_transformer import Linformer
from vit_pytorch import ViT
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def build_model(device: str):
    transformer = Linformer(
        dim=128,
        seq_len=49 + 1,  # 7x7 patches + 1 cls-token
        depth=12,
        heads=8,
        k=64
    )
    vit = ViT(
        dim=128,
        image_size=224,
        patch_size=32,
        num_classes=2,
        transformer=transformer,
        channels=3,
    )
    return vit.to(device)


def get_data_loaders(batch_size: int = 64):
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = dsets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    val_ds = dsets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_model(model: torch.nn.Module, train_loader: DataLoader, device: str, epochs: int = 6):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs.view(-1, 3, 224, 224))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (batch_idx + 1)
        print(f'Epoch : {epoch+1} - loss : {avg_loss} - acc: ? - val_loss : ? - val_acc: ?')


def main():
    device = get_device()
    model = build_model(device)
    train_loader, _ = get_data_loaders(batch_size=64)
    train_model(model, train_loader, device, epochs=6)


if __name__ == '__main__':
    main()