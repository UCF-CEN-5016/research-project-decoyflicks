import torch
from typing import Callable

class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim: int = 10, hidden_dim: int = 5, output_dim: int = 2):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

def train_model(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    epochs: int = 100,
) -> None:
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: Loss = {loss.item()}")

def main() -> None:
    # Build model and optimizer
    ae = Autoencoder()
    optimizer = torch.optim.SGD(ae.parameters(), lr=1e-3)

    # Sample input and target
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 2)

    # Loss function
    loss_function = torch.nn.MSELoss()

    # Training loop
    train_model(ae, inputs, targets, optimizer, loss_function, epochs=100)

    # Check if to_logits.weight is updated (should have zero gradient)
    print("Encoder to_logits.weight gradient:", ae.encoder[0].weight.grad)

if __name__ == "__main__":
    main()