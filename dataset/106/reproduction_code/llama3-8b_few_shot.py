import torch
from vector_quantize_pytorch import VectorQuantizePyTorch

# Initialize vector quantizer
vq = VectorQuantizePyTorch(dim=128, num_embeddings=1024, embedding_dim=8)

# Generate some input data
x = torch.randn(1, 128)

# Compute the distance between x and the embeddings using the buggy code
dist = (x - vq.embeddings) ** 2

print(f"Distance: {torch.sum(dist)}")