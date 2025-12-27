import torch

def normalize_channels(input_tensor):
    # Normalize over the channel dimension
    normalized_shape = [input_tensor.size(1)]
    ln = torch.nn.LayerNorm(normalized_shape)
    return ln(input_tensor)

# Example input tensor with shape (batch_size, channels, height, width)
x = torch.randn(1, 32, 32, 64)  # Shape: (1, 32, 32, 64)

# Apply LayerNorm with correct normalized_shape
y = normalize_channels(x)