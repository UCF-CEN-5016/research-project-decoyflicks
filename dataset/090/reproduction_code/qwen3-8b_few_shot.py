import torch

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
        self.to_logits = torch.nn.Linear(5, 2)  # Not used in forward pass

    def forward(self, x):
        return self.linear(x)  # to_logits is never used in forward pass

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 2)

    def forward(self, x):
        return self.linear(x)

# Build model and optimizer
model = torch.nn.Sequential(Encoder(), Decoder())
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Sample input and target
X = torch.randn(32, 10)
y = torch.randn(32, 2)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss = torch.nn.functional.mse_loss(outputs, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: Loss = {loss.item()}")

# Check if to_logits.weight is updated (should have zero gradient)
print("Encoder to_logits.weight gradient:", model[0].to_logits.weight.grad)