import torch
import torch.nn as nn
import torch.optim as optim

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 10)
        self.bias = nn.Parameter(torch.zeros(10, 10))  # Initializing bias with zeros

    def forward(self, x):
        return self.linear(x) + self.bias

model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)

x = torch.randn(32, 10)
y = torch.randn(32, 10)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(x)
    loss = nn.functional.mse_loss(outputs, y)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}: Loss = {loss.item()}")