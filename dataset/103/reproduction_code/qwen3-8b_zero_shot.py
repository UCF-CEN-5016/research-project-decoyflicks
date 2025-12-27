import torch
from vector_quantize_pytorch import ResidualLFQ

model = ResidualLFQ(
    dim=14,
    codebook_size=100,
    commitment_loss_weight=1.0
)

input = torch.randn(2, 1851, 14)
mask = torch.randint(0, 2, (2, 1851), dtype=torch.bool)
output = model(input, mask=mask)