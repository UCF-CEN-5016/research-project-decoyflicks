import torch
from vector_quantize_pytorch import ResidualSimVQ

torch.manual_seed(42)

batch_size = 1
input_tensor = torch.randn(batch_size, 512, 32, 32)

residual_sim_vq = ResidualSimVQ(
    dim=512,
    num_quantizers=4,
    codebook_size=1024,
    channel_first=True
)

try:
    quantized, indices, commit_loss = residual_sim_vq(input_tensor)
except NameError as e:
    assert str(e) == "name 'return_loss' is not defined"