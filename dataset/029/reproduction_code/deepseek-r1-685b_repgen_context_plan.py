import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

class SimpleModel(torch.nn.Module):
    """
    A basic neural network model with a single fully connected layer.
    """
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = torch.nn.Linear(10, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.fc(x)

def get_device() -> torch.device:
    """
    Determines and returns the appropriate device (MPS, CUDA, or CPU) for computation.
    """
    if torch.backends.mps.is_available():
        print("Using MPS backend.")
        return torch.device('mps')
    elif torch.cuda.is_available():
        print("Using CUDA backend.")
        return torch.device('cuda')
    else:
        print("Using CPU backend.")
        return torch.device('cpu')

def run_training_step(model: torch.nn.Module, data: torch.Tensor, targets: torch.Tensor,
                      optimizer: Adam, criterion: CrossEntropyLoss):
    """
    Executes a single training step: forward pass, loss calculation,
    backward pass, and optimizer step.

    Args:
        model (torch.nn.Module): The neural network model.
        data (torch.Tensor): Input data for the model.
        targets (torch.Tensor): Ground truth labels.
        optimizer (Adam): The optimizer for updating model parameters.
        criterion (CrossEntropyLoss): The loss function.
    """
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    print(f"Training step completed. Loss: {loss.item():.4f}")

def main():
    """
    Main function to set up and run a simple PyTorch training demonstration.
    """
    device = get_device()

    model = SimpleModel().to(device)
    
    dummy_input_data = torch.randn(4, 10).to(device)
    dummy_labels = torch.randint(0, 2, (4,)).to(device)
    
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()
    
    run_training_step(model, dummy_input_data, dummy_labels, optimizer, criterion)

if __name__ == "__main__":
    main()
