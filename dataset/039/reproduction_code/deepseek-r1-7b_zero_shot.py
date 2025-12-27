import torch
from labml_nn.transformers import Rope

# Corrected code slice along multiple dimensions to ensure compatible shapes for broadcasting.
x = torch.randn(2, 3)  # Example input with batch=2 and sequence length=3
model = Rope(...)  # Initialize the model with appropriate parameters.

cos_part = model.cos_cached[:, :, :, :x.shape[0]]
sin_part = model.sin_cached[:, :, :, :x.shape[0]]

# Apply rotation using correctly shaped components.