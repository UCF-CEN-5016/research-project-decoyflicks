import torch
from vector_quantize_pytorch import ResidualVQ

Z_CHANNELS = 512
NUM_QUANTIZERS = 2
CODEBOOK_SIZE = 16 * 1024

model = ResidualVQ(dim=Z_CHANNELS, num_quantizers=NUM_QUANTIZERS,
                   codebook_size=CODEBOOK_SIZE)

input_data = torch.randn(1, Z_CHANNELS)
mask = torch.ones_like(input_data) > 0.5
sampled = torch.randn_like(input_data)

try:
    model.forward(input_data, mask)
except RuntimeError as e:
    print(e)