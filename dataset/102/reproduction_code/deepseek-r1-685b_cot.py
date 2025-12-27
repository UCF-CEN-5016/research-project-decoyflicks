import torch
from vector_quantize_pytorch import ResidualSimVQ

# Setup minimal environment
torch.manual_seed(42)

# Create ResidualSimVQ with problematic settings
quantizer = ResidualSimVQ(
    dim=1024,
    codebook_size=1024,
    num_quantizers=8,
    quantize_dropout=True,  # Required to trigger the bug
    channels_first=True     # Required for shape inconsistency
)

# Create dummy input matching the error case (batch=2, timesteps=17, dim=1024)
x = torch.randn(2, 17, 1024)

# This will trigger both bugs:
# 1. NameError for undefined return_loss
# 2. Shape inconsistencies in the output
quantized, indices, loss = quantizer(x)

# The error occurs during forward pass before reaching these lines
print(quantized.shape)
print([l.shape for l in loss])
print([i.shape for i in indices])