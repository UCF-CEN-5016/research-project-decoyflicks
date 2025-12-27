import torch
from torchvision.models import inception_v3

# Set up environment
batch_size = 4
height = 224
width = 224
channels = 3
num_classes = 1000

# Create random input data
input_data = torch.rand(batch_size, height, width, channels)

# Load the model and extract features
model = inception_v3(num_classes=num_classes)
features = model.features(input_data)

# Calculate a loss function (example: cross-entropy)
labels = torch.randint(0, num_classes, (batch_size,))
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(features, labels)

# Check for NaN values in the loss
assert torch.isnan(loss).any(), "Loss contains NaN values"

# Enable CUDA profiling
torch.cuda.cudnn.benchmark = True

# Assert peak GPU memory usage
peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
assert peak_memory > 1000, f"Peak GPU memory usage is {peak_memory} MB"