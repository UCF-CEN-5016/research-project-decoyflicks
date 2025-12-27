import torch
from torch import nn
from torchvision.transforms import ToTensor

# Define constants
batch_size = 10
image_height = 256
image_width = 256
num_classes = 10

# Create synthetic dataset
dataset = torch.randint(0, 256, (batch_size, image_height, image_width, 3)).float()

# Define a simple CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        return torch.relu(self.conv(x))

model = SimpleCNN(3, num_classes)

# Initialize weights randomly
nn.init.xavier_uniform_(model.conv.weight)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Compute predictions
dataset = ToTensor()(dataset)
output = model(dataset)

# Calculate loss with random labels
labels = torch.randint(0, num_classes, (batch_size,))
loss = criterion(output, labels)

# Check for NaN or infinite values in the loss calculation
assert not torch.isnan(loss).any() and not torch.isinf(loss).any()

# Set up GPU environment if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
dataset = dataset.to(device)

# Monitor GPU memory usage during forward pass
with torch.cuda.amp.autocast():
    output_gpu = model(dataset)

# Assert that GPU memory consumption exceeds a certain threshold indicating an issue
assert torch.cuda.max_memory_allocated(device) > 1024 * 1024  # Minimum of 1MB allocated