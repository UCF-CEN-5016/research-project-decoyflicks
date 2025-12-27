import torch
from vector_quantize_pytorch import ResidualVQ

# Configuration matching the bug report
dim = 512
num_quantizers = 2
codebook_size = 16 * 1024

# Create the ResidualVQ module with problematic settings
vq = ResidualVQ(
    dim=dim,
    num_quantizers=num_quantizers,
    codebook_size=codebook_size,
    stochastic_sample_codes=True,  # Key setting causing the issue
    shared_codebook=True,
    kmeans_init=True,
    threshold_ema_dead_code=2,
    quantize_dropout=True
)

# Simulate training input
batch_size = 32
x = torch.randn(batch_size, dim)

# Simulate multiple training steps to trigger the bug
for i in range(100):
    try:
        # Forward pass that may trigger the error
        quantized, indices, commit_loss = vq(x)
        
        # Backward pass
        loss = torch.mean((quantized - x.detach()) ** 2) + commit_loss
        loss.backward()
        
        print(f"Step {i}: Success")
    except RuntimeError as e:
        print(f"Step {i}: Error - {str(e)}")
        break