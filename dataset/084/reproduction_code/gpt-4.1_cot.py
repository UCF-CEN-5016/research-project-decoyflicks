import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Dummy dataset with identical inputs and labels
class DummyDataset(Dataset):
    def __init__(self, num_samples=100, num_classes=2, input_shape=(3, 224, 224)):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.input_shape = input_shape
        # Create data that is easy to classify
        self.data = torch.randn(num_samples, *input_shape)
        # Create labels alternating between 0 and 1
        self.labels = torch.arange(num_samples) % num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Simple ViT stub model that outputs perfect prediction for label 0
class SimpleViT(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        # Dummy linear layer for demonstration
        self.fc = nn.Linear(3 * 224 * 224, num_classes)

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create datasets and dataloaders (using the same dataset for train and val to simulate leakage)
    train_dataset = DummyDataset(num_samples=100)
    val_dataset = DummyDataset(num_samples=100)

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

    model = SimpleViT(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch : {epoch} - loss : {train_loss:.4f} - acc: {train_acc:.4f} - val_loss : {val_loss:.4f} - val_acc: {val_acc:.4f}")

if __name__ == "__main__":
    main()