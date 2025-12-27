import torch
from vector_quantize_pytorch import ResidualLFQ

# Unstable training setup
lfq = ResidualLFQ(num_embeddings=256, commitment_cost=0.1)

# Sample data
x = torch.randn(2, 1851, 1, 14)  # Input with shape mismatch warning
mask = torch.randint(0, 2, (2, 1851, 1, 14))  # Mask with same shape as x

# Reproduce the bug
loss = lfq(x, mask=mask, commitment_loss_weight=1.0)

print(f"Loss: {loss}")