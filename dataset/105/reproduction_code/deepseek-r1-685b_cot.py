import torch
from vector_quantize_pytorch import ResidualVQ

# Reproducible setup
torch.manual_seed(42)

# Minimal configuration that triggers the bug
vq = ResidualVQ(
    dim=512,
    num_quantizers=2,
    codebook_size=16 * 1024,
    stochastic_sample_codes=True,
    shared_codebook=True,
    threshold_ema_dead_code=2,
    quantize_dropout=True,
    quantize_dropout_cutoff_index=1,
)

# Simulate multi-node behavior by forcing code replacement
with torch.no_grad():
    # Force some codes to be dead
    vq._codebook.cluster_size.data[:1000] = 1  # Mark as barely used
    vq._codebook.embed.data[:1000] = 0  # Make them zero vectors

# Create input that will trigger replacement
x = torch.randn(1, 512, 32, 32)  # Batch of 1 with spatial dims

# Forward pass that may trigger the bug
try:
    quantized, indices, commit_loss = vq(x)
except RuntimeError as e:
    print(f"Error occurred: {e}")
    print("This simulates the race condition where the number of dead codes")
    print("doesn't match the number of sampled replacements during distributed training")