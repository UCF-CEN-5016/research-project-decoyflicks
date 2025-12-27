import torch
import torch.nn as nn
import torch.optim as optim

# Define the Encoder with a to_logits layer that is not used in forward pass
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20)
        self.to_logits = nn.Linear(20, 10)  # This weight is not used in forward

    def forward(self, x):
        x = self.linear(x)
        return x  # to_logits is excluded from forward computation

# Define a simple Decoder that uses the encoder's output
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(20, 10)

    def forward(self, x):
        return self.linear(x)

# Define the full model (Encoder + Decoder)
class Seq2SeqModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Create model, optimizer, and loss function
model = Seq2SeqModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    inputs = torch.randn(32, 10)  # Batch of 32 samples with 10 features
    targets = torch.randint(0, 10, (32, 10))  # Dummy targets
    
    outputs = model(inputs)
    loss = criterion(outputs.view(-1, 10), targets.view(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Check if encoder.to_logits.weight is updated
initial_weight = model.encoder.to_logits.weight.clone()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)  # Ensure gradients are computed
print("Initial encoder.to_logits.weight:", initial_weight)
print("Final encoder.to_logits.weight after training:", model.encoder.to_logits.weight)