import torch
from torch import Tensor
from torch.nn import LayerNorm

def layer_norm_channels(tensor: Tensor) -> Tensor:
    """
    Apply LayerNorm using the size of the channel dimension as the normalized shape.

    Note: This constructs LayerNorm with normalized_shape = [tensor.size(1)]
    and then applies it to the tensor.
    """
    normalized_shape = [tensor.size(1)]
    layer_norm = LayerNorm(normalized_shape)
    return layer_norm(tensor)

# Example input tensor with shape (batch_size, channels, height, width)
x = torch.randn(1, 32, 32, 64)  # Shape: (1, 32, 32, 64)

# Apply LayerNorm with the channel dimension size as normalized_shape
y = layer_norm_channels(x)