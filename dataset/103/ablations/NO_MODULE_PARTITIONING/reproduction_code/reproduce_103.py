import torch
import torch.nn as nn
from vector_quantize_pytorch.lookup_free_quantization import LFQ

torch.manual_seed(42)

batch_size = 2
input_dim = (3700, 14)
input_tensor = torch.randn(batch_size, *input_dim)
mask = torch.ones(2, 1851, 1, 14).bool()

model = LFQ(commitment_loss_weight=1.0)

with torch.no_grad():
    output = model(input_tensor, mask=mask)

# Check for UserWarning
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    output = model(input_tensor, mask=mask)
    assert any("Using a target size" in str(warn.message) for warn in w)

commit_loss = output[2]  # Assuming output contains the loss as the third element
print(commit_loss.shape)