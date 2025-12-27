import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

class Linformer(nn.Module):
    def __init__(self, dim, seq_len, depth, heads, k):
        super(Linformer, self).__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.depth = depth
        self.heads = heads
        self.k = k
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)

class ViT(nn.Module):
    def __init__(self, dim, image_size, patch_size, num_classes, transformer, channels):
        super(ViT, self).__init__()
        self.dim = dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.transformer = transformer
        self.channels = channels
        self.patch_embedding = nn.Linear(channels * patch_size ** 2, dim)
        self.positional_embedding = nn.Parameter(torch.randn(1, (image_size // patch_size) ** 2 + 1, dim))
        self.class_token = nn.Parameter(torch.randn(1, 1, dim))
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        x = x.transpose(1, 2)
        x = self.patch_embedding(x)
        x = x + self.positional_embedding
        x = torch.cat((self.class_token.repeat(x.size(0), 1, 1), x), dim=1)
        x = self.transformer(x)
        x = x[:, 0, :]
        x = self.head(x)
        return x

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set hyperparameters
dim = 128
image_size = 224
patch_size = 32
num_classes = 2
depth = 12
heads = 8
k = 64
batch_size = 32
epochs = 5

# Create dataset and data loader
transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create model, optimizer and loss function
efficient_transformer = Linformer(dim=dim, seq_len=(image_size // patch_size) ** 2 + 1, depth=depth, heads=heads, k=k)
model = ViT(dim=dim, image_size=image_size, patch_size=patch_size, num_classes=num_classes, transformer=efficient_transformer, channels=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(output, 1)
        correct += (predicted == target).sum().item()
    accuracy = correct / len(train_dataset)
    print(f'Epoch : {epoch+1} - loss : {total_loss / len(train_loader)} - acc: {accuracy:.4f}')

    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
    val_accuracy = correct / len(test_dataset)
    print(f'val_loss : {val_loss / len(test_loader)} - val_acc: {val_accuracy:.4f}')