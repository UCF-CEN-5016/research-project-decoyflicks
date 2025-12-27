import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

# Define a simple dataset class
class SimpleDataset(Dataset):
    def __init__(self, transform):
        self.data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Set up data transforms and loaders
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = SimpleDataset(transform)
test_dataset = SimpleDataset(transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define a simple ViT model
class ViT(nn.Module):
    def __init__(self, dim, image_size, patch_size, num_classes, transformer, channels):
        super(ViT, self).__init__()
        self.transformer = transformer
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

# Define the Linformer (efficient transformer)
class Linformer(nn.Module):
    def __init__(self, dim, seq_len, depth, heads, k):
        super(Linformer, self).__init__()
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim, dropout=0.1) for _ in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Initialize the model, device, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
efficient_transformer = Linformer(dim=128, seq_len=49+1, depth=12, heads=8, k=64)
model = ViT(dim=128, image_size=224, patch_size=32, num_classes=2, transformer=efficient_transformer, channels=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(5):
    model.train()
    total_loss = 0
    correct = 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        # Reduce the number of classes to 2 to simulate the issue
        labels = (labels % 2)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()

    accuracy = correct / len(train_loader.dataset)
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}, Accuracy: {accuracy:.4f}')

    # Evaluate the model
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            labels = (labels % 2)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / len(test_loader.dataset)
    print(f'Test Loss: {test_loss / len(test_loader)}, Test Accuracy: {test_accuracy:.4f}')