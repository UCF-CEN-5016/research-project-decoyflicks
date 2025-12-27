import torch
import torch.nn as nn
from vector_quantize_pytorch import ResidualLFQ

torch.manual_seed(42)

batch_size = 2
input_dim = (3700, 14)
input_data = torch.randn(batch_size, *input_dim)
mask = torch.rand(batch_size, 1851, 1, 14) > 0.5

model = ResidualLFQ(dim=14, codebook_size=256, commitment_loss_weight=1.0)

with torch.no_grad():
    output = model(input_data, mask=mask)

print(f"Original input shape: {input_data.shape}")
print(f"Quantized output shape: {output.x.shape}")