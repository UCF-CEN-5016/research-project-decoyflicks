import torch
from vector_quantize_pytorch import ResidualVQ

def create_residual_vq(dim, num_quantizers, codebook_size):
    return ResidualVQ(
        dim=dim,
        num_quantizers=num_quantizers,
        codebook_size=codebook_size,
        stochastic_sample_codes=True,
        shared_codebook=True,
        kmeans_init=True,
        threshold_ema_dead_code=2,
        quantize_dropout=True
    )

def simulate_training(vq, batch_size, num_steps):
    x = torch.randn(batch_size, vq.dim)
    for i in range(num_steps):
        try:
            quantized, indices, commit_loss = vq(x)
            loss = torch.mean((quantized - x.detach()) ** 2) + commit_loss
            loss.backward()
            print(f"Step {i}: Success")
        except RuntimeError as e:
            print(f"Step {i}: Error - {str(e)}")
            break

# Configuration matching the bug report
dim = 512
num_quantizers = 2
codebook_size = 16 * 1024

# Create the ResidualVQ module with problematic settings
vq = create_residual_vq(dim, num_quantizers, codebook_size)

# Simulate training input
batch_size = 32
num_steps = 100

# Simulate multiple training steps to trigger the bug
simulate_training(vq, batch_size, num_steps)