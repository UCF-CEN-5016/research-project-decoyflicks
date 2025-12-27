import torch
from vector_quantize_pytorch import ResidualLFQ

# Setup parameters
batch_size = 2
seq_len = 1851
channels = 14

# Dummy input tensor
x = torch.randn(batch_size, seq_len, channels)

# Mask to simulate padded or inactive positions
mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
mask[0, 1000:] = False  # Mask out some positions in first batch item

# Initialize ResidualLFQ with commitment_loss_weight > 0
lfq = ResidualLFQ(
    dim=channels,
    num_quantizers=1,
    commitment_loss_weight=1.0
)

# Forward pass with mask triggers reshaping and size mismatch warning
out, commit_loss = lfq(x, mask=mask)

print(f"Output shape: {out.shape}")
print(f"Commitment loss: {commit_loss.item()}")