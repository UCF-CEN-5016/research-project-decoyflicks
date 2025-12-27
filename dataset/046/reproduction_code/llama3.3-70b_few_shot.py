import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignmentModel(nn.Module):
    def __init__(self):
        super(AlignmentModel, self).__init__()
        self.fc = nn.Linear(10, 5)  # Simple linear layer

    def forward(self, x):
        x = self.fc(x)
        return x

# Simulate log_probs with incorrect shape (should be 3D)
log_probs = torch.randn(32, 5)  # Incorrect 2D shape

# Create a dummy input
input_length = 10
batch_size = 32
num_classes = 5
x = torch.randn(batch_size, input_length)

# Initialize the model
model = AlignmentModel()

# Forward pass to get the output
output = model(x)

# Try to calculate loss with incorrect log_probs shape
try:
    loss = F.nll_loss(output, torch.argmax(log_probs, dim=-1))
    print(f"Loss: {loss.item()}")
except RuntimeError as e:
    print(f"Error: {e}")