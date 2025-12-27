import torch
import torch.nn as nn
from vector_quantize_pytorch import ResidualVQ

# Minimal environment setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a minimal model configuration
num_embeddings = 10
embedding_dim = 5
num_layers = 2

# Triggering conditions: implicit_neural_codebook=False
model = ResidualVQ(
    num_embeddings=num_embeddings,
    embedding_dim=embedding_dim,
    num_layers=num_layers,
    implicit_neural_codebook=False
)

# Place the model on the device
model.to(device)

# Attempt to verify if MLPs are initialized despite implicit_neural_codebook=False
for name, param in model.named_parameters():
    if "mlp" in name:
        print(f"MLP parameter found: {name}")
        # If this prints, it indicates MLPs are being initialized