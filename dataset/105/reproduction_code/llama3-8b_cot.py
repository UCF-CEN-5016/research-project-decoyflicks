import torch
from vector_quantize_pytorch import ResidualVQ

# Define some constants
Z_CHANNELS = 512
NUM_QUANTIZERS = 2
CODEBOOK_SIZE = 16 * 1024

# Create the ResidualVQ instance
residual_vq = ResidualVQ(dim=Z_CHANNELS, num_quantizers=NUM_QUANTIZERS,
                          codebook_size=CODEBOOK_SIZE)

# Generate some sample data for testing
num_samples = 9330
batch_size = 32
data_shape = (batch_size, Z_CHANNELS)
data = torch.randn(num_samples, *data_shape[-2:])

# Trigger the bug
for i in range(0, num_samples, batch_size):
    z_tilde, _, commit_loss = residual_vq(data[i:i+batch_size])