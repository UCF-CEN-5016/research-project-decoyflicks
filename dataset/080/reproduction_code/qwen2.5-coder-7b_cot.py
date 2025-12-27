import torch
from torch.utils.data import DataLoader
from vit_pytorch.vit_for_small_dataset import ViT, SmallDatasetVit
from typing import Tuple

# Configuration
BATCH_SIZE = 64
PATCH_SIZE = 16
IMAGE_SIZE = 224
NUM_CLASSES = 2
TRAIN_DIR = "path/to/training"
VAL_DIR = "path/to/validation"
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_WORKERS = 4
DROPOUT = 0.1
EMB_DROPOUT = 0.1


def build_vit_model(image_size: int, patch_size: int, num_classes: int) -> ViT:
    return ViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=DROPOUT,
        emb_dropout=EMB_DROPOUT
    )


def prepare_datasets(train_dir: str, val_dir: str, image_size: int, patch_size: int) -> Tuple[SmallDatasetVit, SmallDatasetVit]:
    train_dataset = SmallDatasetVit(train_dir, image_size=image_size, patch_size=patch_size)
    val_dataset = SmallDatasetVit(val_dir, image_size=image_size, patch_size=patch_size)
    return train_dataset, val_dataset


def create_dataloaders(train_dataset: SmallDatasetVit, val_dataset: SmallDatasetVit, batch_size: int
                      ) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    return train_loader, val_loader


def train_one_epoch(model: ViT, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module) -> Tuple[float, int]:
    model.train()
    running_correct = 0
    running_total = 0
    for inputs, labels in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        running_total += labels.size(0)
        running_correct += (predicted == labels).sum().item()

    train_accuracy = running_correct / running_total if running_total else 0.0
    return train_accuracy, running_total


def validate(model: ViT, loader: DataLoader) -> float:
    model.eval()
    val_total = 0
    val_correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    return val_correct / val_total if val_total else 0.0


def main():
    model = build_vit_model(IMAGE_SIZE, PATCH_SIZE, NUM_CLASSES)

    train_dataset, val_dataset = prepare_datasets(TRAIN_DIR, VAL_DIR, IMAGE_SIZE, PATCH_SIZE)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        train_acc, _ = train_one_epoch(model, train_loader, optimizer, criterion)
        val_acc = validate(model, val_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()