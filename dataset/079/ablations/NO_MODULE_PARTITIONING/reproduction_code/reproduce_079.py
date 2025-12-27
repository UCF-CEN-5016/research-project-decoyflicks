import torch
import torch.nn as nn
import torch.nn.functional as F
from vit_pytorch.regionvit import RegionViT

# Define parameters
batch_size = 2
channels = 3
height = 224
width = 224

# Create random input data
input_data = torch.randn(batch_size, channels, height, width)

# Instantiate the RegionViT model
model = RegionViT(dim=(64, 128, 256, 512), depth=(2, 2, 8, 2), window_size=7, num_classes=1000, tokenize_local_3_conv=True, local_patch_size=4, use_peg=False, attn_dropout=0., ff_dropout=0., channels=3)

# Pass the random input data through the model
output = model(input_data)

# Check the shape of the output tensor
assert output.shape == (batch_size, 1000)

# Verify LayerNorm dimension
local_tokens = model.local_encoder(input_data)
assert local_tokens.shape[1] == 64, "LayerNorm input shape mismatch"

# Assert LayerNorm is applied over the wrong dimension
assert local_tokens.shape != (batch_size, 64), "LayerNorm applied over incorrect dimensions"