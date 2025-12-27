import torch
import torch.nn as nn
import torch.nn.functional as F


def build_input(batch_size, n_heads, channels, height, width, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn(batch_size, n_heads, channels, height, width)


def apply_reshape_and_relu(tensor, batch_size, n_heads, channels, height, width):
    # Ensure the tensor has the intended shape
    tensor = tensor.view(batch_size, n_heads, channels, height, width)
    # Non-linear activation step
    tensor = F.relu(tensor)
    # Flatten all dims except channels so LayerNorm can be applied per-channel
    tensor = tensor.view(-1, channels)
    # Another activation to show the flow
    tensor = F.relu(tensor)
    return tensor


def main():
    batch_size = 1
    n_heads = 4
    channels = 64
    height = 10
    width = 10

    x = build_input(batch_size, n_heads, channels, height, width)
    x = apply_reshape_and_relu(x, batch_size, n_heads, channels, height, width)

    layer_norm = nn.LayerNorm(channels)

    try:
        x = layer_norm(x)
    except RuntimeError as e:
        print(f"LayerNorm error: {e}")


if __name__ == "__main__":
    main()