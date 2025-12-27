# reproduce_bug.py

import torch
from labml import labml
from labml.utils.logger import logger
from labml_nn.optimizers import AdamFP16, GradScalerFP16
from torch.nn.modules.loss import CrossEntropyLoss

# Define a model architecture that uses positional embeddings (e.g., Transformer-based model)
class PositionalEmbeddingModel(torch.nn.Module):
    def __init__(self, embedding_dim, seq_len):
        super(PositionalEmbeddingModel, self).__init__()
        self.embedding = torch.nn.Embedding(seq_len, embedding_dim)

    def forward(self, x):
        pos_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        return self.embedding(pos_ids) + x

# Initialize the positional embedding layer with learnable parameters
embedding_dim = 512
seq_len = 1024
model = PositionalEmbeddingModel(embedding_dim, seq_len)

# Set the batch size to 32 and sequence length to 1024
batch_size = 32
x = torch.randn(batch_size, seq_len, embedding_dim, device='cuda')

# Simulate forward pass through the model, including the positional embedding layer
output = model(x)

# Calculate a loss function (e.g., CrossEntropyLoss) on the output
criterion = CrossEntropyLoss()
target = torch.randint(0, 10, (batch_size, seq_len), device='cuda')
loss = criterion(output.view(-1, embedding_dim), target.view(-1))

# Observe and record the values of the loss tensor before backpropagation
logger.info(f"Initial Loss: {loss.item()}")

# Simulate backward pass through the model to perform gradient updates
optimizer = AdamFP16(model.parameters())
scaler = GradScalerFP16()
scaled_loss = scaler.scale(loss)
scaled_loss.backward()

# Verify that the loss value contains NaNs after multiple iterations
for _ in range(10):
    optimizer.step()
    scaler.update()

    # Calculate new loss
    output = model(x)
    loss = criterion(output.view(-1, embedding_dim), target.view(-1))
    
    # Check for NaNs in the loss tensor
    if torch.isnan(loss).any():
        logger.info("NaN found in loss tensor")
        break

logger.info(f"Final Loss: {loss.item()}")