import torch
from torch import nn, optim
from typing import Iterable, Callable

# Model definition
class SpatialTransformNetwork(nn.Module):
    """
    Minimal spatial transform network similar to the original example.
    Architecture and forward pass preserved to maintain original behavior.
    """
    def __init__(self) -> None:
        super().__init__()
        # convolutional block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layer kept with original dimensions (preserved behavior)
        self.fc1 = nn.Linear(7 * 7 * 64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.pool(self.conv1(x)))
        # Flatten and apply linear layer (preserved original view shape)
        return self.fc1(x.view(-1, 7 * 7 * 32))


# Training loop
def train_network(
    model: nn.Module,
    dataloader: Iterable,
    optimizer: optim.Optimizer,
    criterion: Callable,
    epochs: int = 5,
) -> None:
    """
    Train the provided model using the given dataloader, optimizer, and loss.
    Core training behavior preserved from the original implementation.
    """
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Preserve original gradient clipping call
            nn.utils.clip_grad_norm(optimizer.param_groups, 1.0)
            loss.backward()
            optimizer.step()