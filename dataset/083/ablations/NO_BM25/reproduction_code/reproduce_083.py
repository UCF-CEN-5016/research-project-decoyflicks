import torch
import torch.nn as nn
from vit_pytorch.cross_vit import CrossViT

torch.manual_seed(42)

batch_size = 4
height, width = 224, 224
input_data = torch.randn(batch_size, 3, height, width)

model = CrossViT(
    image_size=224,
    num_classes=10,
    sm_dim=128,
    lg_dim=256,
    sm_patch_size=12,
    lg_patch_size=16,
    sm_enc_depth=1,
    lg_enc_depth=4,
    cross_attn_depth=2,
    dropout=0.1,
    emb_dropout=0.1
)

output_logits = model(input_data)

assert output_logits.shape == (batch_size, 10)
assert not torch.isnan(output_logits).any()

print(output_logits)

problematic_input = torch.zeros(batch_size, 3, height, width)
output_from_problematic_input = model(problematic_input)

assert not torch.all(output_from_problematic_input == output_logits)