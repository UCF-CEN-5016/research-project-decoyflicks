import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

# Create an input tensor with shape (8910, 3)
input_tensor = torch.randn(8910, 3)

# Instantiate the model with input size 1 and output size 1
model = MyModel(3, 1)

try:
    output = model(input_tensor)
    print("Output shape:", output.shape)
except RuntimeError as e:
    print("Error occurred:", e)

# Reshape the input tensor
input_tensor = input_tensor[:, 0].unsqueeze(1)

output = model(input_tensor)

# Change the model input size to 3 and output size to 1
model = MyModel(3, 1)