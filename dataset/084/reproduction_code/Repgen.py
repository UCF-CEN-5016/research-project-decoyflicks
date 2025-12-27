import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, RandomHorizontalFlip, Normalize
from torch.utils.data import DataLoader
from models.vit import ViT

# Define model parameters
dim = 128
seq_len = 49 + 1  # 7x7 patches + 1 cls-token
depth = 12
heads = 8
k = 64
image_size = 224
patch_size = 32
num_classes = 10
channels = 3

# Setup dataset and dataloader
transform = ToTensor() + RandomHorizontalFlip() + Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
val_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and move to device
model = ViT(dim=dim, seq_len=seq_len, depth=depth, heads=heads, k=k, image_size=image_size, patch_size=patch_size, num_classes=num_classes, channels=channels).to(device)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    model.train()
    train_loss = 0.0
    correct_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct_train += (predicted == labels).sum().item()

    train_accuracy = correct_train / len(train_dataset)

    model.eval()
    val_loss = 0.0
    correct_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_val += (predicted == labels).sum().item()

    val_accuracy = correct_val / len(val_dataset)

    print(f'Epoch {epoch + 1}, Train Loss: {train_loss / len(train_dataset)}, Train Accuracy: {train_accuracy:.2f}, Val Loss: {val_loss / len(val_dataset)}, Val Accuracy: {val_accuracy:.2f}')

# Assert validation accuracy
assert val_accuracy == 1.0, 'Validation accuracy is not 100%'