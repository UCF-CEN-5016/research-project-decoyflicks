import torch
from vector_quantize_pytorch import ResidualLFQ

x = torch.randn(2, 1851, 14)
mask = torch.ones(2, 1851, dtype=torch.bool)
mask[:, 1850] = False

vq = ResidualLFQ(
    dim=14,
    num_quantizers=1,
    num_codevectors=256,
    commitment_loss_weight=1.0
)

vq(x, mask=mask)