import torch
import torch.nn as nn
from vector_quantize_pytorch import ResidualVQ

# Set up ResidualVQ with implicit_neural_codebook=False
vq = ResidualVQ(
    dim=128,
    num_tokens=128,
    num_layers=2,
    implicit_neural_codebook=False  # Should not initialize MLPs
)

# Check if MLPs are initialized
for name, module in vq.named_modules():
    if isinstance(module, nn.MultiLayerPerceptron):
        print(f"MLP initialized: {name}")
        # If this prints, it means the MLPs are being initialized even when implicit_neural_codebook=False