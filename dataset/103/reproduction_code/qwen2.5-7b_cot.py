import torch
from vector_quantize_pytorch import ResidualLFQ

# Set random seed for reproducibility
torch.manual_seed(42)

# Create sample input tensor
input_tensor = torch.randn(2, 1851, 14)

# Create a mask tensor
mask = torch.ones(2, 1851, dtype=torch.bool)

# Initialize ResidualLFQ model
model = ResidualLFQ(
    dim=14,
    codebook_size=256,
    codebook_dim=14,
    commitment_loss_weight=1.0
)

# Disable gradient computation for the forward pass
with torch.no_grad():
    output = model(input_tensor, mask=mask)

# The commitment loss calculation triggers the shape mismatch warning