import torch

# Example input tensor with shape (batch_size, channels, height, width)
x = torch.randn(1, 32, 32, 64)  # Shape: (1, 32, 32, 64)

# Correct normalized_shape for channel-wise normalization
ln = torch.nn.LayerNorm([32])  # Normalize over the channel dimension

# Apply LayerNorm
y = ln(x)

import torch

# Example input tensor with shape (batch_size, channels, height, width)
x = torch.randn(1, 32, 32, 64)  # Shape: (1, 32, 32, 64)

# Incorrect normalized_shape (does not match the input shape)
ln = torch.nn.LayerNorm([64, 32, 32])  # Mismatched shape

# Apply LayerNorm
y = ln(x)