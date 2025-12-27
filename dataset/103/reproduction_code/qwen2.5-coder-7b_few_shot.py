import torch
from vector_quantize_pytorch import ResidualLFQ
from typing import Tuple


def create_sample_input(batch: int = 2, channels: int = 3, height: int = 14, width: int = 14) -> torch.Tensor:
    """Create a random input tensor with given dimensions."""
    return torch.randn(batch, channels, height, width)


def create_sample_mask(batch: int = 2, height: int = 14, width: int = 14) -> torch.Tensor:
    """Create a random boolean mask for the spatial dimensions."""
    return torch.randint(0, 2, (batch, height, width)).bool()


def build_residual_lfq(dim: int = 3, codebook_size: int = 10, commitment_loss_weight: float = 1.0) -> ResidualLFQ:
    """Initialize and return a ResidualLFQ instance."""
    return ResidualLFQ(
        dim=dim,
        codebook_size=codebook_size,
        commitment_loss_weight=commitment_loss_weight
    )


def run_inference(model: ResidualLFQ, input_tensor: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """Perform a forward pass through the model with an optional mask."""
    return model(input_tensor, mask=mask)


def main() -> None:
    # Prepare sample data
    sample_input = create_sample_input()
    sample_mask = create_sample_mask()

    # Initialize model
    quantizer = build_residual_lfq(dim=3, codebook_size=10, commitment_loss_weight=1.0)

    # Forward pass
    output = run_inference(quantizer, sample_input, mask=sample_mask)

    # Print output shape (illustrative)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    main()