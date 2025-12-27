import torch
from torch.utils.checkpoint import checkpoint

class NestedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1024, 1024)

    def forward(self, x):
        # Unbind the nested tensor
        tensors = torch.unbind(x)
        # Apply linear to each tensor
        processed = [self.linear(t) for t in tensors]
        # Stack back into nested tensor
        result = torch.nested.nested_tensor(processed)
        return result

model = NestedModel()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Create a nested tensor with two elements, each 5x1024
nested_input = torch.nested.nested_tensor([
    torch.randn(5, 1024),
    torch.randn(5, 1024)
])

# Forward pass
output = model(nested_input)

# Loss is the sum of all elements
loss = output.sum()

# Backward pass
loss.backward()

import torch
from torch.utils.checkpoint import checkpoint

class NestedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1024, 1024)

    def forward(self, x):
        # Unbind the nested tensor
        tensors = torch.unbind(x)
        # Apply linear to each tensor
        processed = [self.linear(t) for t in tensors]
        # Stack back into nested tensor
        result = torch.nested.nested_tensor(processed)
        return result

model = NestedModel()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Create a nested tensor with two elements, each of shape (5, 1024)
nested_input = torch.nested.nested_tensor([
    torch.randn(5, 1024),
    torch.randn(5, 1024)
])

# Forward pass
output = model(nested_input)

# Loss is the sum of all elements
loss = output.sum()

# Backward pass
loss.backward()

import torch
from torch.utils.checkpoint import checkpoint

class NestedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1024, 1024)

    def forward(self, x):
        # Unbind the nested tensor
        tensors = torch.unbind(x)
        # Apply linear to each tensor
        processed = [self.linear(t) for t in tensors]
        # Stack back into nested tensor
        result = torch.nested.nested_tensor(processed)
        return result

model = NestedModel()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Create a nested tensor with two elements, each of shape (5, 1024)
nested_input = torch.nested.nested_tensor([
    torch.randn(5, 1024),
    torch.randn(5, 1024)
])

# Forward pass
output = model(nested_input)

# Loss is the sum of all elements
loss = output.sum()

# Backward pass
loss.backward()