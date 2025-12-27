import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Simplified Linformer
class Linformer(nn.Module):
    def __init__(self, dim, seq_len, depth, heads, k):
        super(Linformer, self).__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.depth = depth
        self.heads = heads
        self.k = k
        # Simplified linear layer for demonstration
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        # Simplified forward pass
        return self.linear(x)

# Simplified ViT
class ViT(nn.Module):
    def __init__(self, dim, image_size, patch_size, num_classes, transformer, channels):
        super(ViT, self).__init__()
        self.dim = dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.transformer = transformer
        self.channels = channels
        # Simplified classification head
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        # Simplified forward pass
        x = self.transformer(x)
        x = self.head(x)
        return x

# Dummy dataset for demonstration
class DummyDataset(Dataset):
    def __init__(self, size, num_classes):
        self.size = size
        self.num_classes = num_classes
        self.data = np.random.rand(size, 128)  # Simplified data
        self.labels = np.random.randint(0, num_classes, size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            'data': torch.from_numpy(self.data[idx]).float(),
            'label': torch.tensor(self.labels[idx])
        }

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and dataset
efficient_transformer = Linformer(
    dim=128,
    seq_len=50,  # Simplified sequence length
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

dataset = DummyDataset(313, 2)  # Dummy dataset
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch in dataloader:
        data = batch['data'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}, Accuracy: {accuracy}')