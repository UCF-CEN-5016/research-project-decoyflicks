import torch
from torch import nn
from torch.utils.data.distributed import DataParallel

class EncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        # Unused parameter that isn't part of any computation graph
        self.to_logits = nn.Linear(hidden_size, 1)  # weight not used elsewhere
        self.decoder = nn.Linear(1, input_size)

    def forward(self, x):
        encoded = self.encoder(x)
        return F.relu(self.decoder(encoded))

def train(model, optimizer, criterion, train_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(model.device)
            target = target.to(model.device)

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Example usage without DDP for simplicity:
batch_size = 32
input_size = 10
hidden_size = 50

encoder = nn.Linear(input_size, hidden_size)
to_logits = nn.Linear(hidden_size, 1)  # Unused layer but tracked by PyTorch

model = EncoderDecoder(input_size, hidden_size)

# Dummy data and loader (simplified example without DDP)
x = torch.randn(batch_size, input_size).to('cuda')
y = x @ torch.ones(batch_size, 1).to('cuda')

train_loader = [ (x, y) ]

# Using a single worker for simplicity
model_single = nn.DataParallel(model)
optimizer = torch.optim.SGD(model_single.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Train the model
train(model_single, optimizer, criterion, train_loader, epochs=5)

# Check if to_logits.weight has changed (should remain unchanged)
print("Initial weights:", list(to_logits.parameters()))