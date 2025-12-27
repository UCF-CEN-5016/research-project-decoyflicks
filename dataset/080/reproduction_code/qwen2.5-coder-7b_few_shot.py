import torch
from torch.utils.data import TensorDataset, DataLoader
from vit_pytorch import ViT

def make_dataloader(samples: torch.Tensor, targets: torch.Tensor, batch_size: int) -> DataLoader:
    dataset = TensorDataset(samples, targets)
    return DataLoader(dataset, batch_size=batch_size)

def prepare_data(num_samples: int = 100,
                 image_shape=(224, 224, 3),
                 num_classes: int = 2,
                 val_size: int = 50,
                 batch_size: int = 32):
    images = torch.randn(num_samples, *image_shape)
    labels = torch.randint(0, num_classes, (num_samples,))
    train_images, val_images = images, images[:val_size]
    train_labels, val_labels = labels, labels[:val_size]
    train_loader = make_dataloader(train_images, train_labels, batch_size=batch_size)
    val_loader = make_dataloader(val_images, val_labels, batch_size=batch_size)
    return train_loader, val_loader, train_labels, val_labels

def build_model(image_size: int = 224,
                patch_size: int = 16,
                num_classes: int = 2,
                dim: int = 1024,
                depth: int = 6,
                heads: int = 16,
                mlp_dim: int = 2048,
                dropout: float = 0.1,
                emb_dropout: float = 0.1):
    return ViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        emb_dropout=emb_dropout
    )

def run_training(num_epochs: int = 10, batch_size: int = 32):
    train_loader, val_loader, _, val_labels = prepare_data(batch_size=batch_size)
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        for batch_images, batch_targets in train_loader:
            outputs = model(batch_images)
            loss = torch.nn.functional.cross_entropy(outputs, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_correct = 0
        for val_images, val_targets in val_loader:
            outputs = model(val_images)
            val_correct += (outputs.argmax(dim=1) == val_targets).sum().item()

        val_acc = val_correct / len(val_labels)
        print(f"Epoch {epoch} - Validation Accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    run_training()