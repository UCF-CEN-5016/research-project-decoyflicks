import torch
from vector_quantize_pytorch import ResidualLFQ

# Initialize LFQ model
model = ResidualLFQ(
    dim=14,
    codebook_size=128,
    num_res_layers=1,
    commitment_loss_weight=1.0
)

# Create a mask
mask = torch.ones(2, 1851, 1)

# Create a random input tensor
input_tensor = torch.randn(3700, 14)

# Forward pass with mask
output = model(input_tensor, mask=mask)

# This will cause a shape mismatch and incorrect commit loss
print("Output shape:", output.shape)