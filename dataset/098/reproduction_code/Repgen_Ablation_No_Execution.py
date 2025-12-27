import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator

class CustomDataset(Dataset):
    def __init__(self, length, batch_size):
        self.length = length
        self.batch_size = batch_size
        self.data = torch.randn(length, batch_size, 64)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx]

# Instantiate the custom dataset
dataset = CustomDataset(1000, 32)

# Define the model architecture
model = torch.nn.Sequential(
    torch.nn.Conv1d(64, 128, kernel_size=3),
    torch.nn.GELU(),
    torch.nn.GroupNorm(num_groups=32, num_channels=128),
    torch.nn.Conv1d(128, 64, kernel_size=3),
    torch.nn.GELU(),
    torch.nn.GroupNorm(num_groups=32, num_channels=64)
)

# Move the model to the appropriate device
device = Accelerator().device
model.to(device)

# Create a DataLoader object
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define an optimizer instance
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = torch.mean((outputs - data) ** 2)
        loss.backward()
        optimizer.step()
        if torch.isnan(loss):
            raise ValueError("NaN encountered during training")