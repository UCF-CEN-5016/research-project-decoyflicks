import torch
from torch.utils.data import TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence

# Create a list of tensors representing a nested tensor
x = [torch.randn(2, 1024) for _ in range(5)]  # Each tensor has shape (2, 1024)

# Convert the list into a packed sequence (nested tensor)
nested_tensor = pack_padded_sequence(x, [2] * len(x), batch_first=True, enforce_sorted=False)

# Define a simple model that processes the nested tensor
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1024, 2)

    def forward(self, x):
        # Process each element of the nested tensor
        processed = []
        for t in x:
            processed.append(self.linear(t))  # Apply linear transformation
        # Sum all processed elements
        return torch.stack(processed).sum()

# Initialize the model
model = SimpleModel()

# Forward pass
output = model(nested_tensor)

# Compute loss
loss = output.sum()

# Backward pass
loss.backward()