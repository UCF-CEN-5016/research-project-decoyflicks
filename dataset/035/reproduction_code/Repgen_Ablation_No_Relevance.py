import torch
from labml import section
from labml_helpers.device_info import DeviceInfo
from labml_nn.optimizers.adam import MyAdam
from labml_nn.optimizers.mnist_experiment import Model

# Set batch size and input dimension
batch_size = 64
input_dim = (28, 28)

# Create random input data
input_data = torch.randn(batch_size, 1, *input_dim)
device = DeviceInfo().cuda if DeviceInfo().has_cuda else DeviceInfo().cpu
input_data = input_data.to(device)

# Define target tensor
target = torch.ones(batch_size, dtype=torch.long).to(device)

# Create model instance
model = Model().to(device)

# Initialize optimizers
my_adam = MyAdam(model.parameters())
torch_adam = torch.optim.Adam(model.parameters())

# Calculate loss
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(model(input_data), target)

# Perform backward pass
loss.backward()

# Warmup steps
for _ in range(100):
    my_adam.step()
    torch_adam.step()

# Assert expected error
try:
    # Intentionally cause a size mismatch by using incorrect input dimensions
    incorrect_input = torch.randn(batch_size, 2, *input_dim)
    model(incorrect_input)
except RuntimeError as e:
    assert isinstance(e, RuntimeError) and 'size mismatch' in str(e), 'Expected size mismatch error due to feature count mismatch.'