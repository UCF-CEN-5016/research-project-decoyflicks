import torch
from vector_quantize_pytorch import ResidualVQ

# Initialize a ResidualVQ model with a large codebook size
vq = ResidualVQ(
    dim=512,
    num_quantizers=2,
    codebook_size=16 * 1024,
    stochastic_sample_codes=True,
    shared_codebook=True,
    commitment_weight=1.0,
    kmeans_init=True,
    threshold_ema_dead_code=2,
    quantize_dropout=True,
    quantize_dropout_cutoff_index=1,
    quantize_dropout_multiple_of=1,
)

# Dummy input for the model
x = torch.randn(1, 512)

# Simulate a scenario where the codebook is accessed and modified
# For demonstration, we create a fake codebook of size 9331
codebook = torch.randn(9331, 512)

# Create indices and a mask that would cause a shape mismatch
ind = torch.arange(9331)  # Shape (9331,)
mask = torch.tensor([True] * 9330)  # Shape (9330,)
sampled = torch.randn(9330, 512)  # Shape (9330, 512)

# Simulate the assignment that would cause the error
try:
    # Attempt to assign the sampled tensor to the codebook using the mask
    codebook[ind][mask] = sampled
except RuntimeError as e:
    print("Error occurred due to shape mismatch:")
    print(e)