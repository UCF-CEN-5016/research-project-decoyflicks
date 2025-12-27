import torch
import torch.nn as nn
import torch.nn.functional as F
from vit_pytorch.regionvit import RegionViT

batch_size = 2
height, width, channels = 224, 224, 3
input_data = torch.rand(batch_size, channels, height, width)

model = RegionViT(dim=(64, 128, 256, 512), depth=(2, 2, 8, 2), window_size=7, num_classes=1000, tokenize_local_3_conv=True, local_patch_size=4, use_peg=False, attn_dropout=0., ff_dropout=0.)

output = model(input_data)
assert output.shape == (batch_size, 1000)

local_tokens = model.local_encoder[0](input_data)
assert local_tokens.shape == (batch_size, 64, height // 2, width // 2)

assert torch.isnan(output).any()