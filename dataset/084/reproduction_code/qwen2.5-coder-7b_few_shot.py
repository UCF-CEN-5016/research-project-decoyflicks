import torch
from torch.utils.data import DataLoader, TensorDataset
from efficientvit.models import ViT, Linformer
from torch import nn, optim
from typing import Tuple


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dataloaders(batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    # Create dummy dataset (all samples are the same)
    inputs = torch.randn(32, 3, 224, 224)  # Dummy images
    targets = torch.zeros(32, dtype=torch.long)  # All samples labeled as class 0

    dataset = TensorDataset(inputs, targets)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def build_model(device: torch.device) -> nn.Module:
    linformer = Linformer(
        dim=128,
        seq_len=49 + 1,  # 7x7 patches + 1 cls-token
        depth=12,
        heads=8,
        k=64,
    )
    vit_model = ViT(
        dim=128,
        image_size=224,
        patch_size=32,
        num_classes=2,
        transformer=linformer,
        channels=3,
    ).to(device)
    return vit_model


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, loss_fn: nn.Module, device: torch.device) -> None:
    model.train()
    for batch_inputs, batch_targets in loader:
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = loss_fn(outputs, batch_targets)
        loss.backward()
        optimizer.step()


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            outputs = model(batch_inputs)
            total_loss += loss_fn(outputs, batch_targets).item()
            _, preds = torch.max(outputs, 1)
            total += batch_targets.size(0)
            correct += (preds == batch_targets).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def main() -> None:
    epochs = 5
    batch_size = 32
    device = get_device()

    train_loader, val_loader = create_dataloaders(batch_size=batch_size)
    model = build_model(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch+1} - loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")


if __name__ == "__main__":
    main()