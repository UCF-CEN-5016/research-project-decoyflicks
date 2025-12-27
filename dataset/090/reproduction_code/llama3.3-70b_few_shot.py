import torch
import torch.nn as nn
import torch.distributed as dist

# Define a simple Encoder/Decoder model
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.to_logits = nn.Linear(10, 10)  # This weight doesn't update

    def forward(self, x):
        # This line is missing, causing the weight to not be updated
        # return self.to_logits(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

# Initialize the model, optimizer, and loss function
model = nn.Sequential(Encoder(), Decoder())
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Sample data
X = torch.randn(32, 10)
y = torch.randn(32, 10)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: Loss = {loss.item()}")

    # Check if the weight is being updated
    print("Encoder to_logits weight:", model[0].to_logits.weight.mean())