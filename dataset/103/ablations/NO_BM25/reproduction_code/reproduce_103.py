import torch
import torch.nn as nn
from vector_quantize_pytorch.lookup_free_quantization import LFQ

torch.manual_seed(42)

batch_size = 2
input_dim = (3700, 14)
input_tensor = torch.randn(batch_size, *input_dim)
mask = torch.ones(2, 1851, 1, 14, dtype=torch.bool)

model = LFQ(dim=input_dim[1], commitment_loss_weight=1.0)

with torch.no_grad():
    output = model(input_tensor, mask=mask)

print(output)