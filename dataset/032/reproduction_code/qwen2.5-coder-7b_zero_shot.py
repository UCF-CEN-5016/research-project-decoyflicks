import torch
import torch.nn.functional as F

def approx_gelu(x: torch.Tensor) -> torch.Tensor:
    """Compute the GELU activation using PyTorch's approximate implementation."""
    return F.gelu(x, approximate=True)

def main() -> None:
    input_tensor = torch.tensor([1.0, 2.0])
    print(approx_gelu(input_tensor))

if __name__ == "__main__":
    main()