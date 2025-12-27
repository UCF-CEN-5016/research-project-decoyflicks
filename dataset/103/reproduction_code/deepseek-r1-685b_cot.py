import torch
from vector_quantize_pytorch import ResidualLFQ

# Set up minimal environment
torch.manual_seed(42)
batch_size = 2
seq_len = 3700  # Large enough to trigger the issue
dim = 14

# Create random input and mask (approximately half masked)
x = torch.randn(batch_size, seq_len, dim)
mask = torch.randint(0, 2, (batch_size, seq_len)).bool()

# Initialize quantizer with commitment loss
quantizer = ResidualLFQ(
    dim=dim,
    codebook_size=256,
    num_quantizers=1,  # Single quantizer for minimal case
    commitment_loss_weight=1.0  # Must be >0 to trigger
)

# Forward pass with mask - this will trigger the warning
quantized, indices, commit_loss = quantizer(x, mask=mask)

# Expected shapes:
# original_input should match quantized shape for proper commit loss
# But gets reshaped differently due to mask handling
print(f"Input shape: {x.shape}")
print(f"Quantized shape: {quantized.shape}")
print(f"Commit loss shape mismatch warning should appear above")