import torch
from vector_quantize_pytorch import ResidualVQ

# Create ResidualVQ with implicit_neural_codebed=False
model = ResidualVQ(
    num_embeddings=512,
    embedding_dim=64,
    num_codebooks=2,
    implicit_neural_codebook=False
)

# Check if MLPs are initialized (should not be)
print("MLP exists" if hasattr(model, 'mlp') else "MLP not found")

import torch
from vector_quantize_pytorch import ResidualVQ

# Create ResidualVQ with implicit_neural_codebook=False
model = ResidualVQ(
    num_embeddings=512,
    embedding_dim=64,
    num_codebooks=2,
    implicit_neural_codebook=False
)

# Check if MLPs are initialized (should not be)
print("MLP exists" if hasattr(model, 'mlp') else "MLP not found")