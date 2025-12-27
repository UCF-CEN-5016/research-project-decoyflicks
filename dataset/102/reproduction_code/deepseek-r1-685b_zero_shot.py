import torch
from vector_quantize_pytorch import ResidualSimVQ

quantizer = ResidualSimVQ(
    dim=1024,
    codebook_size=1024,
    heads=8,
    quantize_dropout=True,
    channels_first=True
)

x = torch.randn(2, 1024, 17)
quantizer(x)