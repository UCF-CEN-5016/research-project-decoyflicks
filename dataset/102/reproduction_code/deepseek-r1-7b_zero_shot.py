import torch
from vector_quantize_pytorch import VectorQuantize, ResidualSimVQ

# Initialize VectorQuantize with necessary parameters (e.g., codebook size)
vq = VectorQuantize(
    num_embeddings=1024,
    embedding_dim=64,
    straighten=False,
    unique=True
)

# Create an instance of ResidualSimVQ, ensuring it's initialized correctly without relying on undefined variables
model = ResidualSimVQ(
    vq=vq,
    # Add any other necessary parameters as required by the model
    # For this example, we set quantize_dropout to True based on prior usage
    quantize_dropout=True
)

# The forward pass should now run without the NameError
input_tensor = torch.randn(17, 64)  # Example input tensor
output = model(input_tensor)
print(output.shape)

import torch
from vector_quantize_pytorch import VectorQuantize, ResidualSimVQ

# Initialize VectorQuantize with necessary parameters (e.g., codebook size)
vq = VectorQuantize(num_embeddings=1024, embedding_dim=64, straighten=False, unique=True)

# Create an instance of ResidualSimVQ
model = ResidualSimVQ(
    vq=vq,
    quantize_dropout=True
)

# Example input tensor (shape: [batch_size, sequence_length, feature_dim])
input_tensor = torch.randn(17, 64)
output = model(input_tensor)
print(f"Output shape: {output.shape}")