import torch
import torch.nn as nn
from vector_quantize_pytorch.lookup_free_quantization import LFQ

torch.manual_seed(42)

batch_size = 2
input_dim = (3700, 14)
input_data = torch.randn(batch_size, *input_dim)

model = LFQ(dim=14, codebook_size=16, commitment_loss_weight=1.0)

mask = torch.randint(0, 2, (batch_size, 1851), dtype=torch.bool)

output = model(input_data, mask=mask)

original_input_shape = output[0].shape
quantized_shape = output[1].shape

print(f"Original Input Shape: {original_input_shape}")
print(f"Quantized Shape: {quantized_shape}")

# Check for UserWarning
try:
    commit_loss = output[2]
except UserWarning as e:
    print(e)