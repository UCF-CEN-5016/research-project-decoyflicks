import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CustomModel(nn.Module):
    def __init__(self) -> None:
        super(CustomModel, self).__init__()
        self.linear = nn.Linear(10, 10)
        self.add_bias = nn.Parameter(torch.zeros(10, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.add_bias

def train_model(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor,
                optimizer: optim.Optimizer, epochs: int = 100) -> None:
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.mse_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: Loss = {loss.item()}")

if __name__ == "__main__":
    model = CustomModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    x = torch.randn(32, 10)
    y = torch.randn(32, 10)

    train_model(model, x, y, optimizer, epochs=100)