import torch
from vector_quantize_pytorch import ResidualLFQ

dim = 14
quantizer = ResidualLFQ(
    dim=dim,
    codebook_size=256,
    commitment_loss_weight=1.0
)

batch_size = 2
seq_len = 1851
x = torch.randn(batch_size, seq_len, dim)
mask = torch.ones(batch_size, seq_len).bool()

quantized, indices, commit_loss = quantizer(x, mask=mask)