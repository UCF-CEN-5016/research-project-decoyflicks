import torch
import torch.nn as nn

# Minimal setup: define input tensor shape and LayerNorm with possibly wrong normalized_shape

# Suppose input tensor shape is (batch_size, channels, height, width)
B, C, H, W = 2, 3, 4, 4

# Create a random input tensor with shape (B, C, H, W)
x = torch.randn(B, C, H, W)

# Incorrect LayerNorm usage:
# The bug is using LayerNorm with normalized_shape = C (channels) but input is (B, C, H, W)
# LayerNorm expects the normalized_shape to match trailing dimensions of input.

# Let's simulate the problematic LayerNorm initialization from the bug report
# normalized_shape = C  # This is wrong if input is (B, C, H, W) and not permuted

layer_norm = nn.LayerNorm(normalized_shape=C)

try:
    # This should raise an error because input shape and normalized_shape don't match
    out = layer_norm(x)
except Exception as e:
    print("Error encountered:", e)

# Correct usage would be either:
# 1) Permute x to (B, H, W, C) then LayerNorm with normalized_shape=C
x_permuted = x.permute(0, 2, 3, 1)  # (B, H, W, C)
layer_norm_correct = nn.LayerNorm(normalized_shape=C)
out_correct = layer_norm_correct(x_permuted)
print("Correct usage output shape:", out_correct.shape)

# 2) Or use normalized_shape = (C, H, W) if keeping input as (B, C, H, W)
layer_norm_spatial = nn.LayerNorm(normalized_shape=(C, H, W))
out_spatial = layer_norm_spatial(x)
print("Spatial normalization output shape:", out_spatial.shape)