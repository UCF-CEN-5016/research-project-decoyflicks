import torch
from torch.nn.functional import gelu

def demonstrate_gelu_type_error():
    """Create a random tensor and call gelu with an invalid 'approximate' argument to trigger TypeError."""
    tensor = torch.randn(2, 3)
    try:
        gelu(tensor, approximate=True)
    except TypeError as exc:
        print(f"Caught error: {exc}")

if __name__ == "__main__":
    demonstrate_gelu_type_error()