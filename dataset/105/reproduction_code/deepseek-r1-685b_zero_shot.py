import torch
from vector_quantize_pytorch import ResidualVQ

torch.manual_seed(42)

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

x = torch.randn(1, 512, 32, 32)

# Simulate multiple forward passes to trigger code expiration
for _ in range(1000):
    try:
        z_tilde, _, _ = vq(x)
    except RuntimeError as e:
        if "shape mismatch" in str(e):
            print("Bug reproduced!")
            break