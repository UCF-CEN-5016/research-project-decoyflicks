import torch
from vector_quantize_pytorch import ResidualLFQ

# Setup LFQ with mask and commitment loss
quantizer = ResidualLFQ(
    dim=14,
    codebook_size=256,
    num_quantizers=2,
    commitment_loss_weight=1.0  # Important for reproducing the bug
)

# Create input and mask
batch_size = 2
seq_len = 1851
dim = 14

x = torch.randn(batch_size, seq_len, dim)
mask = torch.ones(batch_size, seq_len).bool()  # Simple all-true mask

# Forward pass that triggers the bug
quantized, indices, commit_loss = quantizer(x, mask=mask)

# This will produce:
# 1. User warning about shape mismatch
# 2. Incorrect commit loss calculation
print(f"Commit loss: {commit_loss}")
print(f"Quantized shape: {quantized.shape}")
print(f"Original input shape: {x.shape}")