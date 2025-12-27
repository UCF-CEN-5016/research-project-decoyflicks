import torch
import torch.nn as nn
from torch.optim import SGD

# Define a small model with layers prone to instability
model = nn.Sequential(
    nn.Linear(50, 10),
    nn.ReLU(),
    nn.Linear(10)
)

# Create input and target data scaled appropriately for stability
X = torch.randn(32, 50) / (torch.sqrt(torch.tensor(50.0)))
y = torch.randn(32, 1).requires_grad_()

# Use a small learning rate to attempt stable training
optimizer = SGD(model.parameters(), lr=1e-4)

for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X)
    loss = nn.functional.mse_loss(outputs, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: Loss = {loss.item()}")