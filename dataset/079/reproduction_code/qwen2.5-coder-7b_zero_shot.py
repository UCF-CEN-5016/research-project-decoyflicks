import torch
from vit_pytorch import RegionViT

model = RegionViT()

def _trigger_bug(module, inp):
    # Intentionally apply LayerNorm with a single integer normalized_shape to a 4D tensor.
    # This will trigger a dimension mismatch similar to the reported RegionViT local token embedding bug.
    ln = torch.nn.LayerNorm(64)
    ln(inp[0])

model.register_forward_pre_hook(_trigger_bug)

input_ = torch.randn(1, 3, 224, 224)
output = model(input_)