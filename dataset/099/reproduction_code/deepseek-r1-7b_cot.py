import torch
from torch import nn
from torch.utils.data import DataLoader

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define model architecture with layers prone to FP8 issues (e.g., scaled inputs)
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=2048):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, 10)  # MNIST has 10 classes
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# Initialize model and move to device
model = SimpleTransformer()
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# Prepare dataset (simulated for illustration)
def create_fake_dataset(batch_size=32):
    # Simulate inputs that might stress FP8 precision
    inputs = torch.randn(batch_size, 768).to(device) * 1000  # Large values to test FP8 limits
    targets = torch.randint(0, 10, (batch_size, )).to(device)
    return inputs, targets

# Data loader
dataset = [(torch.randn(768), torch.randint(0,10,(1,))) for _ in range(32)]
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop with NaN handling
nan_loss_triggered = False

for epoch in range(1):
    for inputs, targets in dataloader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if torch.isnan(loss):
            print("Loss became NaN. Training stopped.")
            break
            
    else:
        continue  # Only executed if loop wasn't broken
    break

print(f"Training completed without NaN loss: {not nan_loss_triggered}")