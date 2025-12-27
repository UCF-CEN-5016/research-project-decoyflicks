import torch
from vector_quantize_pytorch import ResidualLFQ

# Set up minimal environment
input_shape = (2, 1851, 1, 14)
mask_shape = (3700, 14)

# Create a simple ResidualLFQ model with mask and commitment_loss_weight > 0
model = ResidualLFQ(num_embeddings=512, num_heads=8, dropout=0.1, commitment_loss_weight=1.0)
model.init_weights()

# Generate input data
input_data = torch.randn(*input_shape)

# Create a mask tensor with the specified shape
mask = torch.zeros(*mask_shape).bool()

# Wrap everything in a forward pass
output = model.forward(input_data, mask)

print(output.shape)  # Should print the original shape before reshaping

# Check if there's any warning or error