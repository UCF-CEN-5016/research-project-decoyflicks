import torch
import torch.nn as nn

# Define a simple model that expects input with 1 feature
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # Linear layer expecting input of shape (batch_size, 1)

    def forward(self, x):
        return self.linear(x)

# Create an input tensor with shape (8910, 3) — this is the actual input
input_tensor = torch.randn(8910, 3)  # shape (batch_size, 3)

# Instantiate the model
model = MyModel()

# Attempt to pass the input tensor to the model
try:
    output = model(input_tensor)
    print("Output shape:", output.shape)
except RuntimeError as e:
    print("Error occurred:", e)

input_tensor = input_tensor.reshape(8910, 1)  # shape (batch_size, 1)
output = model(input_tensor)

model = nn.Linear(3, 1)  # Change the input size to 3