import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import ViTForImageClassification, ViTConfig

# Ensure deterministic behavior
torch.manual_seed(42)
np.random.seed(42)


class SimpleImageDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        # Expect images in HWC (N, H, W, C); convert to CHW for PyTorch
        images_chw = np.transpose(images.astype(np.float32), (0, 3, 1, 2))
        self.images = torch.from_numpy(images_chw)
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def generate_synthetic_dataset(num_examples: int = 100, image_size: int = 224, num_classes: int = 2):
    X = np.random.randn(num_examples, image_size, image_size, 3)  # HWC
    y = np.random.randint(0, num_classes, (num_examples,))
    return SimpleImageDataset(X, y)


# Data loader creation
TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 32


def build_dataloaders(dataset: Dataset, train_batch_size: int = TRAIN_BATCH_SIZE, val_batch_size: int = VAL_BATCH_SIZE):
    train_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader


def initialize_vit_model(device: torch.device, num_classes: int = 2):
    config = ViTConfig(
        image_size=224,
        patch_size=32,
        hidden_size=128,            # embed_dim
        num_hidden_layers=12,       # depth
        num_attention_heads=8,      # num_heads
        intermediate_size=128 * 4,  # mlp_ratio * embed_dim
        qkv_bias=True,
        hidden_dropout_prob=0.1,
        num_labels=num_classes
    )
    model = ViTForImageClassification(config).to(device)
    return model


def train_epoch(model: torch.nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device):
    model.train()
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()


def validate_epoch(model: torch.nn.Module, val_loader: DataLoader, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total if total > 0 else 0.0


def main():
    device = torch.device('cuda')

    dataset = generate_synthetic_dataset()
    train_loader, val_loader = build_dataloaders(dataset)

    model = initialize_vit_model(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, step_size_up=1000)

    num_epochs = 5
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}')
        train_epoch(model, train_loader, optimizer, device)
        val_acc = validate_epoch(model, val_loader, device)
        print(f'Validation Accuracy: {val_acc:.4f}')

        # Early exit if validation accuracy hits 100%
        if val_acc >= 1.0:
            break


if __name__ == "__main__":
    main()