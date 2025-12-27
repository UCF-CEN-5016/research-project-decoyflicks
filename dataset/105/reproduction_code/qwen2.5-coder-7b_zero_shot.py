import torch
from vector_quantize_pytorch import ResidualVQ
from typing import Tuple


Z_DIM = 512
NUM_QUANTIZERS = 2
CODEBOOK_SIZE = 16 * 1024


def build_residual_vq(dim: int, num_quantizers: int, codebook_size: int) -> ResidualVQ:
    """Construct and return a ResidualVQ model with given configuration."""
    return ResidualVQ(dim=dim, num_quantizers=num_quantizers, codebook_size=codebook_size)


def prepare_inputs(z_dim: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a random input tensor, a boolean mask, and a sampled tensor matching the input shape."""
    input_tensor = torch.randn(1, z_dim)
    mask_tensor = torch.ones_like(input_tensor) > 0.5
    sampled_tensor = torch.randn_like(input_tensor)
    return input_tensor, mask_tensor, sampled_tensor


def run_forward(model: ResidualVQ, input_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> None:
    """Invoke the model's forward pass and print any RuntimeError that occurs."""
    try:
        model.forward(input_tensor, mask_tensor)
    except RuntimeError as err:
        print(err)


def main() -> None:
    model = build_residual_vq(Z_DIM, NUM_QUANTIZERS, CODEBOOK_SIZE)
    input_tensor, mask_tensor, _sampled = prepare_inputs(Z_DIM)
    run_forward(model, input_tensor, mask_tensor)


if __name__ == "__main__":
    main()