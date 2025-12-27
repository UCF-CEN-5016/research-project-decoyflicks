import torch
import torch.nn as nn
from vector_quantize_pytorch import ResidualVQ

# Set up minimal environment
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define ResidualVQ parameters
Z_CHANNELS = 512
NUM_QUANTIZERS = 2
CODEBOOK_SIZE = 16 * 1024

# Initialize ResidualVQ
vq = ResidualVQ(
    dim=Z_CHANNELS,
    num_quantizers=NUM_QUANTIZERS,
    codebook_size=CODEBOOK_SIZE,
    stochastic_sample_codes=True,
    shared_codebook=True,
    commitment_weight=1.0,
    kmeans_init=True,
    threshold_ema_dead_code=2,
    quantize_dropout=True,
    quantize_dropout_cutoff_index=1,
    quantize_dropout_multiple_of=1,
).to(device)

# Create dummy input tensor
input_tensor = torch.randn(1, Z_CHANNELS, 16, 16).to(device)

# Call the forward method of ResidualVQ
try:
    output = vq(input_tensor)
except RuntimeError as e:
    print(f"Error: {e}")