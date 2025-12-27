import torch
from vector_quantize_pytorch import ResidualVQ

# Configuration matching the bug report
dim = 512
num_quantizers = 2
codebook_size = 16 * 1024

def create_vq_module(dim, num_quantizers, codebook_size):
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

def simulate_training(vq, num_steps):
    batch_size = 32
    x = torch.randn(batch_size, dim)

    for i in range(num_steps):
        try:
            quantized, indices, commit_loss = vq(x)

            loss = torch.mean((quantized - x.detach()) ** 2) + commit_loss
            loss.backward()

            print(f"Step {i}: Success")
        except RuntimeError as e:
            print(f"Step {i}: Error - {str(e)}")
            break

vq = create_vq_module(dim, num_quantizers, codebook_size)
simulate_training(vq, 100)