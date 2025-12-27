import torch
from vector_quantize_pytorch import ResidualVQ

# Create ResidualVQ with implicit_neural_codebook set to False
model = ResidualVQ(num_embeddings=512, embedding_dim=64, num_codebooks=2, implicit_neural_codebook=False)

# Check if MLPs are initialized
mlp_status = "MLP exists" if hasattr(model, 'mlp') else "MLP not found"
print(mlp_status)