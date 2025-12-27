import torch

class Autoencoder(torch.nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(10, 5)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(5, 2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Build model and optimizer
model = Autoencoder()
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
print("Encoder to_logits.weight gradient:", model.encoder[0].weight.grad)