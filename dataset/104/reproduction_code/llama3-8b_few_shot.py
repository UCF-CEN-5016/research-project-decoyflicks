import torch
from vector_quantize_pytorch import ResidualVQ

# Minimal setup for the bug
vq = ResidualVQ(dim=128, depth=2)

print("MLP initialized?", vq.mlp_initialized)