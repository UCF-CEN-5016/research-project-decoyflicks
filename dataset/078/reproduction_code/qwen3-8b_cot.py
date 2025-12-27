import torch
import torch.nn as nn
import torch.nn.functional as F

# Minimal setup to reproduce the bug
class NestedTensorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1024, 1024)
    
    def forward(self, x):
        # Simulate nested tensor processing (unbinding along dim=1)
        # This mimics the problematic line in the original code
        unbinded = torch.unbind(x, dim=1)
        processed = [self.linear(tensor) for tensor in unbinded]
        return torch.stack(processed, dim=1)

# Create a 3D nested tensor with shape [5, 2, 1024]
# This shape is critical for triggering the gradient mismatch
nested_tensor = torch.randn(5, 2, 1024, requires_grad=True)

# Initialize the model
model = NestedTensorModel()

# Forward pass
output = model(nested_tensor)

# Compute loss (arbitrary)
loss = output.sum()

# Backward pass (this will trigger the error)
loss.backward()