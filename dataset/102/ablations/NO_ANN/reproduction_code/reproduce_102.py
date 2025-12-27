import torch
from vector_quantize_pytorch import ResidualSimVQ

torch.manual_seed(42)
batch_size = 1
input_dim = (512, 32, 32)

x = torch.randn(batch_size, *input_dim)
residual_sim_vq = ResidualSimVQ(dim=512, num_quantizers=4, codebook_size=1024, channel_first=True)

try:
    quantized, indices, commit_loss = residual_sim_vq(x)
except NameError as e:
    print(e)  # Expected output: NameError: name 'return_loss' is not defined