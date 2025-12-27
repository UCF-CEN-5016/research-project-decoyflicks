import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the gelu activation function with approximate=True
class MyGELU(nn.Module):
    def forward(self, x):
        return F.gelu(x, approximate='True')  # This should be a string

model = MyGELU()
input_data = torch.randn(1)
output = model(input_data)

print("Output:", output)