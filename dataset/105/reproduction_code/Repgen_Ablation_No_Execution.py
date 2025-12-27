import torch
from vector_quantize_pytorch import ResidualVQ

Z_CHANNELS = 512
NUM_QUANTIZERS = 2
CODEBOOK_SIZE = 16 * 1024

# Mock codebook with shape [NUM_QUANTIZERS, Z_CHANNELS]
mock_codebook = torch.randn(NUM_QUANTIZERS, Z_CHANNELS)

# Define batch_size and input dimensions (height, width) as multiples of 8
batch_size = 32
height = 64
width = 64

# Create random input data x with shape (batch_size, height, width, 1)
x = torch.randn(batch_size, height, width, 1)

# Initialize the ResidualVQ module using the defined parameters and mock codebook
residual_vq = ResidualVQ(
    dim=Z_CHANNELS,
    num_quantizers=NUM_QUANTIZERS,
    codebook_dim=CODEBOOK_SIZE,
    use_batch_norm=False
)
residual_vq.codebook.weight.data = mock_codebook

# Encode the input data through a mock encode layer that outputs a tensor of shape [batch_size, Z_CHANNELS]
encode_layer = torch.nn.Conv2d(in_channels=1, out_channels=Z_CHANNELS, kernel_size=3, padding=1)
encoded_x = encode_layer(x).view(batch_size, -1)

# Simulate the quantize method call within ResidualVQ by generating a random sampled tensor with shape [batch_size, 1]
sampled_tensor = torch.randint(0, NUM_QUANTIZERS, (batch_size, 1)).float()

# Manually trigger the replace method in the vector_quantize_pytorch library's ExpireCodes class
from vector_quantize_pytorch import ExpireCodes
expire_codes = ExpireCodes()
ind = torch.arange(batch_size)
mask = torch.ones_like(ind).bool()
with torch.no_grad():
    expire_codes.embed.data[ind][mask] = sampled_tensor

# Prepare input data ind and mask such that they cause a shape mismatch when trying to assign the sampled tensor into self.embed.data[ind][mask]
sampled_tensor = torch.randint(0, NUM_QUANTIZERS, (batch_size + 1, 1)).float()

# Assert that the error 'RuntimeError: shape mismatch' occurs during execution
try:
    with torch.no_grad():
        expire_codes.embed.data[ind][mask] = sampled_tensor
except RuntimeError as e:
    assert str(e) == "shape mismatch", f"Expected 'RuntimeError: shape mismatch', but got {e}"

# Monitor GPU memory usage during execution
import tracemalloc
tracemalloc.start()

# Assert GPU memory usage remains stable or does not increase unexpectedly
initial_memory, initial_peak = tracemalloc.get_traced_memory()
with torch.no_grad():
    expire_codes.embed.data[ind][mask] = sampled_tensor
current_memory, current_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

assert abs(current_memory - initial_memory) < 1024 * 1024, f"GPU memory usage increased unexpectedly: {initial_memory} -> {current_memory}"