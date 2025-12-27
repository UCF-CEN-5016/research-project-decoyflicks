import torch
from vector_quantize_pytorch import ResidualLFQ

# Set up minimal environment
torch.manual_seed(42)

# Create sample input with shape (batch, seq_len, features)
input_tensor = torch.randn(2, 1851, 14)  # Shape: (2, 1851, 14)

# Create a mask tensor (same shape as input's sequence dimension)
mask = torch.ones(2, 1851, dtype=torch.bool)  # All True for simplicity

# Initialize ResidualLFQ with commitment loss
model = ResidualLFQ(
    dim=14,
    codebook_size=256,
    codebook_dim=14,
    commitment_loss_weight=1.0  # Trigger commit loss calculation
)

# Forward pass with mask
with torch.no_grad():
    output = model(input_tensor, mask=mask)

# The commit loss calculation will trigger the shape mismatch warning