import torch
from vector_quantize_pytorch.lookup_free_quantization import ResidualLFQ

# Configuration
DEVICE = 'cpu'
BATCH_SIZE = 1
IN_CHANNELS = 32
PATCH_SIZE = (4, 4)  # kept for symmetry with original code (unused)
MASK_RATIO = 0.5


def create_input(batch_size: int, in_channels: int, device: str) -> torch.Tensor:
    """Create a random input tensor on the specified device."""
    return torch.randn(batch_size, in_channels, 28, 28).to(device)


def create_flat_mask(tensor: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """Create a flattened binary mask tensor matching the input tensor's batch and spatial/channel size."""
    batch = tensor.size(0)
    # Create mask with same shape as tensor excluding batch, then flatten per batch
    mask_shape = (batch, *tensor.shape[1:])
    mask = torch.bernoulli(torch.ones(mask_shape) * mask_ratio).to(tensor.device)
    return mask.view(mask.size(0), -1)


def build_model() -> ResidualLFQ:
    """Initialize the ResidualLFQ model with the parameters that reproduce the original behavior."""
    return ResidualLFQ(
        in_channels=32,
        out_channels=64,
        embedding_dim=16,
        nll_loss=True,
        use_momentum_for_mean=False,
        commit_loss_weight=1.0,
    )


def main():
    original_input = create_input(BATCH_SIZE, IN_CHANNELS, DEVICE)
    mask = create_flat_mask(original_input, MASK_RATIO)

    model = build_model()

    try:
        output = model(original_input, mask=mask)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()