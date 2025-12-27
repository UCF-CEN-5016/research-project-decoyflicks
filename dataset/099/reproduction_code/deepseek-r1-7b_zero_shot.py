import torch
from transformer_engine import TE
import math

# Initialize model parameters
input_size = 512
hidden_size = 64
output_size = 512
batch_size = 32
num_layers = 2

# Create a simple model with two layers using TransformerEngine kernels for FP8
class SimpleModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        # Using FP16 LinearBackward kernels as an example; adjust according to TE's FP8 settings
        self.linear1 = TELinearBackward(input_size, hidden_size)
        self.linear2 = TELinearBackward(hidden_size, output_size)

    def forward(self, x):
        return self.linear2(self.linear1(x))

# Example optimizer setup for FP8 precision; adjust parameters as needed
def create_optimizer(model):
    # Placeholder for actual optimizer configuration specific to TransformerEngine and FP8
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    return optimizer

model = SimpleModel(input_size, hidden_size, output_size).cuda()
optimizer = create_optimizer(model)

# Define a simple loss function (adjust according to your model task)
def criterion(input, target):
    return torch.nn.functional.mse_loss(input, target)

# Example data
x = torch.randn(batch_size, input_size).to(torch.device('cuda'))
y = torch.randn(batch_size, output_size).to(torch.device('cuda'))

# Training loop with NaN detection and handling
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    
    # Check for NaN loss
    if math.isnan(loss.item()):
        print(f"NaN loss encountered at epoch {epoch}")
        break
    
    loss.backward()
    optimizer.step()

print("Training completed without NaN loss")