import torch
from typing import Callable, Tuple
from x_transformers import Attend

def generate_alibi_pos(pos) -> torch.Tensor:
    """Custom alibi position function that returns a 4D tensor.
    The input parameter `pos` is accepted for compatibility but not used.
    """
    return torch.randn(2, 3, 4, 5)  # Example 4D shape

def build_input_tensors(shape: Tuple[int, int, int, int] = (2, 3, 4, 5)) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create sample query and key tensors with the given shape."""
    query = torch.randn(*shape)
    key = torch.randn(*shape)
    return query, key

def initialize_attend_module(dim: int = 5, alibi_pos: Callable = None, attn_flash: bool = True) -> Attend:
    """Initialize and return an Attend module with the provided configuration."""
    return Attend(dim=dim, alibi_pos=alibi_pos, attn_flash=attn_flash)

def run_demo():
    """Build inputs, initialize Attend, and compute the attention output."""
    query, key = build_input_tensors()
    attend_module = initialize_attend_module(alibi_pos=generate_alibi_pos)
    output = attend_module(query, key)
    return output

if __name__ == "__main__":
    run_demo()