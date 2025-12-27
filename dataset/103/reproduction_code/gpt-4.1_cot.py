import torch
from vector_quantize_pytorch import ResidualLFQ

# Set seed for reproducibility
torch.manual_seed(0)

# Parameters
batch_size = 2
seq_len = 20
embed_dim = 14

# Create dummy input tensor [batch, seq_len, embed_dim]
x = torch.randn(batch_size, seq_len, embed_dim)

# Create a mask with some tokens masked out (False)
# For example, mask out half tokens in second batch element
mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
mask[1, 10:] = False  # Mask last 10 tokens in batch 1

# Instantiate ResidualLFQ with commitment_loss_weight > 0
lfq = ResidualLFQ(
    dim=embed_dim,
    num_quantizers=2,
    num_embeddings=512,
    commitment_loss_weight=1.0,
)

# Run forward pass with mask
out, commit_loss = lfq(x, mask=mask)

print("Output shape:", out.shape)
print("Commitment loss:", commit_loss)