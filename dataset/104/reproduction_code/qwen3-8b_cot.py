import torch
from vector_quantize_pytorch import ResidualVQ

# Minimal setup to reproduce the bug
model = ResidualVQ(
    dim=64,
    num_codevectors=128,
    codebook_size=256,
    implicit_neural_codebook=False  # Trigger condition: flag is False
)

# Check if MLP parameters are present (should not be)
print("Model parameters (should not include MLPs):")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# Optional: Check for MLP layers in the model structure
print("\nModel structure (should not contain MLP layers):")
print(model)