import torch
import torch.nn as nn
from vit_pytorch.regionvit import RegionViT

# Parameters
batch_size = 2
channels = 3
height = 224
width = 224

# Create random input data
input_data = torch.randn(batch_size, channels, height, width)

# Instantiate the RegionViT class
model = RegionViT(dim=(64, 128, 256, 512), depth=(2, 2, 8, 2), window_size=7, num_classes=1000, tokenize_local_3_conv=True, local_patch_size=4, use_peg=False, attn_dropout=0., ff_dropout=0.)

# Pass the input data through the local_encoder
local_tokens = model.local_encoder(input_data)

# Pass the input data through the region_encoder
region_tokens = model.region_encoder(input_data)

# Iterate through the layers
for down, peg, transformer in model.layers:
    local_tokens = down(local_tokens)
    region_tokens = down(region_tokens)
    local_tokens = peg(local_tokens)
    local_tokens, region_tokens = transformer(local_tokens, region_tokens)

# Call the to_logits method
logits = model.to_logits(region_tokens)

# Check output shape
print(logits.shape)

# Assert conditions
assert local_tokens.shape == (2, 64, 56, 56)
assert region_tokens.shape == (2, 256, 32, 32)
assert logits.shape == (2, 1000)
assert not torch.isnan(logits).any()