import torch
from vector_quantize_pytorch import ResidualSimVQ

# Create a ResidualSimVQ model
model = ResidualSimVQ(num_embeddings=256, embedding_dim=128, num_heads=8)

# Train the model (this will raise an error)
losses = []
for _ in range(10):
    inputs = torch.randn(1, 17, 1024)  # Batch size 1
    output = model(inputs)
    loss = output.mean()
    losses.append(loss)
print(f"Losses: {losses}")