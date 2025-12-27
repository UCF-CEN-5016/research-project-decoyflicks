import torch
from vector_quantize_pytorch import ResidualLFQ, get_codebook

# Create a random input and codebook
input_data = torch.randn(3700, 14)
codebook, _ = get_codebook(num_embeddings=256, dim=14)

# Create the LFQ model with mask and commitment loss weight
model = ResidualLFQ(dim=14, num_heads=8, num_layers=2, codebook=codebook, use_mask=True, commitment_loss_weight=1.0)

# Run forward pass with mask
output = model(input_data)

print(output.shape)