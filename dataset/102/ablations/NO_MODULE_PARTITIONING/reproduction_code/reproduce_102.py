import torch
from vector_quantize_pytorch import ResidualSimVQ

def test_residual_sim_vq():
    residual_sim_vq = ResidualSimVQ(
        dim=512,
        num_quantizers=4,
        codebook_size=1024,
        channel_first=True
    )

    x = torch.randn(1, 512, 32, 32)
    try:
        quantized, indices, commit_loss = residual_sim_vq(x)
    except NameError as e:
        print(e)

test_residual_sim_vq()