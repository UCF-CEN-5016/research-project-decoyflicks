import torch
from vector_quantize_pytorch import ResidualVQ

dim = 8
codebook_size = 16
num_quantizers = 4

model = ResidualVQ(
    dim=dim,
    codebook_size=codebook_size,
    num_quantizers=num_quantizers,
    implicit_neural_codebook=False
)

x = torch.randn(1, 32, dim)
out = model(x)

for name, param in model.named_parameters():
    if 'mlp' in name:
        print(f"MLP parameter found: {name}")