import torch
import torch.nn as nn
import torch.optim as optim
from transformer_engine.pytorch import Linear  # TE Linear layer
from transformer_engine.pytorch.fp8 import fp8_autocast

# Minimal transformer block using TE Linear layer
class MinimalTransformerBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear1 = Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

def main():
    # Check cuda availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This example requires CUDA device.")

    torch.manual_seed(0)

    # Hyperparameters
    batch_size = 4
    seq_len = 8
    hidden_dim = 16
    lr = 1e-3

    # Instantiate model and optimizer
    model = MinimalTransformerBlock(hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Dummy data (random)
    input_data = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    target = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    model.train()

    for step in range(10):
        optimizer.zero_grad()
        # Enable FP8 autocast context
        with fp8_autocast():
            output = model(input_data)
            loss = loss_fn(output, target)

        print(f"Step {step}, Loss: {loss.item()}")

        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    main()