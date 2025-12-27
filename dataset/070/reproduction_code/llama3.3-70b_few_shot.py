import torch
import deepspeed

# Define a simple model
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = torch.nn.Linear(5, 5)

    def forward(self, x):
        x = self.fc(x)
        # Simulate all_reduce operation
        handle = deepspeed.comm.all_reduce(x)
        return x

# Initialize deepspeed
deepspeed.init_distributed()

# Create a model and move it to the GPU
model = SimpleModel()
model.to(torch.device("cuda:0"))

# Create a random input tensor
input_tensor = torch.randn(5, 5).to(torch.device("cuda:0"))

# Run the model
try:
    output = model(input_tensor)
except AttributeError as e:
    print(f"Error: {e}")