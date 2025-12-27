import torch
import torch.nn as nn
import torch.optim as optim

# Define a model with an uninitialized bias
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.bias = nn.Parameter(torch.Tensor(10, 10))  # Uninitialized bias

    def forward(self, x):
        return self.linear(x) + self.bias

# Initialize the model
model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Generate random input and target
x = torch.randn(32, 10)
y = torch.randn(32, 10)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(x)
    loss = torch.nn.functional.mse_loss(outputs, y)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}: Loss = {loss.item()}")

self.bias = nn.Parameter(torch.zeros(10, 10))