import torch
from vector_quantize_pytorch import ResidualLFQ

# Initialize the model
model = ResidualLFQ(dim=14, codebook_size=100, commitment_loss_weight=1.0)

# Generate random input and mask
input_data = torch.randn(2, 1851, 14)
mask = torch.randint(0, 2, (2, 1851), dtype=torch.bool)

# Process the input through the model
output = model(input_data, mask=mask)