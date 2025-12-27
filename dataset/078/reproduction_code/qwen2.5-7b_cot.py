import torch
import torch.nn as nn

class NestedTensorModel(nn.Module):
    def __init__(self):
        super(NestedTensorModel, self).__init__()
        self.linear = nn.Linear(1024, 1024)
    
    def forward(self, x):
        # Process each nested tensor separately
        processed = torch.stack([self.linear(tensor) for tensor in x.unbind(dim=1)], dim=1)
        return processed

# Create a 3D nested tensor with shape [5, 2, 1024]
nested_tensor = torch.randn(5, 2, 1024, requires_grad=True)

# Initialize the model
model = NestedTensorModel()

# Forward pass
output = model(nested_tensor)

# Compute loss (arbitrary)
loss = output.sum()

# Backward pass
loss.backward()