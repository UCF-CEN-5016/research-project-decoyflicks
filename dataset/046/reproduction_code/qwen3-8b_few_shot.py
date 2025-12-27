import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)

# Create a model that outputs 2-D (batch_size, num_classes)
model = SimpleModel(input_dim=10, num_classes=50)
input_data = torch.randn(32, 10)  # (batch_size=32, input_dim=10)

# Simulate log_probs output (incorrect shape)
log_probs = model(input_data)  # Shape: (32, 50)

# Force check for 3-D shape (as expected by forced alignment code)
if log_probs.dim() != 3:
    raise RuntimeError("log_probs must be 3-D (batch_size, input length, num classes)")