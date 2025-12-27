import torch
from torch import nn
from torch.utils.data import DataLoader

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 2048, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

def create_fake_dataset(batch_size: int = 32, input_dim: int = 768):
    # Simulate inputs that might stress FP8 precision (large magnitude)
    inputs = torch.randn(batch_size, input_dim).to(get_device()) * 1000
    targets = torch.randint(0, 10, (batch_size,)).to(get_device())
    return inputs, targets

def build_dataloader(num_samples: int = 32):
    # Keep the original sample construction style (per-sample tensors)
    samples = [(torch.randn(768), torch.randint(0, 10, (1,))) for _ in range(num_samples)]
    return DataLoader(samples, batch_size=num_samples, shuffle=True)

def train_one_epoch(model: nn.Module, dataloader: DataLoader, loss_fn, optimizer, device: str):
    nan_loss_triggered = False
    for inputs, targets in dataloader:
        # Move batch to device and ensure targets have correct shape
        inputs = inputs.to(device)
        targets = targets.to(device).squeeze()

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if torch.isnan(loss):
            print("Loss became NaN. Training stopped.")
            nan_loss_triggered = True
            break
    return nan_loss_triggered

# Setup
device = get_device()
model = SimpleTransformer().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# Data loader (kept consistent with original code style)
dataloader = build_dataloader(num_samples=32)

# Training loop with NaN handling (single epoch to match original behavior)
nan_loss_triggered = False
for _ in range(1):
    nan_loss_triggered = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
    if nan_loss_triggered:
        break

print(f"Training completed without NaN loss: {not nan_loss_triggered}")