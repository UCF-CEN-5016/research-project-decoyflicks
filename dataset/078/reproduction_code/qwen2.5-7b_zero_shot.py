import torch
from torch.utils.data import TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence

class SimpleModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        processed = self.linear(x)
        return processed.sum()

# Create a list of tensors representing a nested tensor
x = [torch.randn(2, 1024) for _ in range(5)]  # Each tensor has shape (2, 1024)
nested_tensor = torch.stack(x)

# Define the model and initialize it
model = SimpleModel(input_size=1024, output_size=2)

# Forward pass
output = model(nested_tensor)

# Compute loss
loss = output.sum()

# Backward pass
loss.backward()