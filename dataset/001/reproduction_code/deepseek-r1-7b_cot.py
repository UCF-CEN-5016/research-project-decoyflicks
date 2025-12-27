import torch
from torch import nn, optim

# Minimal model setup adjusted from cluster_example
class SpatialTransformNetwork(nn.Module):
    def __init__(self):
        super(SpatialTransformNetwork, self).__init__()
        # Use efficient layers and activation functions to prevent NaNs
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(7*7*64, 10)

    def forward(self, x):
        x = self.relu(self.maxpool(self.conv1(x)))
        # Flatten and apply linear layer
        return self.fc1(x.view(-1, 7*7*32))

# Training setup
def train_model(network, dataloader, optimizer, criterion, epochs=5):
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, targets)
            # Add gradient clipping if necessary
            nn.utils.clip_grad_norm(optimizer.param_groups, 1.0)
            loss.backward()
            optimizer.step()

# Ensure memory efficient operations (e.g., using FP16 where applicable)