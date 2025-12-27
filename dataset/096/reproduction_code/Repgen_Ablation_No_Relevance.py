import torch
from torchvision.transforms.functional import gaussian_blur

# Set batch size and image dimensions
batch_size = 32
height, width = 256, 256

# Create random uniform input data with shape (batch_size, height, width, 3)
input_data = torch.rand((batch_size, height, width, 3))

# Apply Gaussian Blur to the input data using the default parameters
blurred_data = gaussian_blur(input_data, kernel_size=(5, 5), sigma=1.5)

# Verify output contains NaN values in loss calculation
loss = ((blurred_data - blurred_data.mean()) ** 2).mean()
print("Loss:", loss.item())

# Monitor GPU memory usage during execution
import torch.cuda as cuda
memory_allocated = cuda.memory_allocated()
print("GPU Memory Allocated:", memory_allocated)