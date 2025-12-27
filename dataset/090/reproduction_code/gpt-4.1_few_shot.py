import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.to_logits = nn.Linear(10, 10)
        # Note: to_logits is NOT used in forward

    def forward(self, x):
        return self.linear(x)

model = Encoder()
optimizer = optim.SGD(model.parameters(), lr=1e-2)

input = torch.randn(5, 10)
target = torch.randn(5, 10)

for epoch in range(3):
    optimizer.zero_grad()
    output = model(input)
    loss = nn.functional.mse_loss(output, target)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch} loss: {loss.item()}")

# Check if to_logits.weight gradient is None (it will be)
print("to_logits.weight grad:", model.to_logits.weight.grad)