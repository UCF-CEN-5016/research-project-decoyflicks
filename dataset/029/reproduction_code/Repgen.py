import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

# Simple model for testing
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = torch.nn.Linear(10, 2)
        
    def forward(self, x):
        return self.fc(x)

# Entry point function
def main():
    # Try to access MPS backend directly to trigger the error
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # Create model and move to device
    model = SimpleModel().to(device)
    
    # Create dummy data
    x = torch.randn(4, 10).to(device)
    y = torch.randint(0, 2, (4,)).to(device)
    
    # Set up optimizer and loss
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()
    
    # Training step
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

if __name__ == "__main__":
    main()