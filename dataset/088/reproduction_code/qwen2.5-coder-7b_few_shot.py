import torch
import torch.nn.functional as F
from typing import Any

def pad_tensor_right(tensor: torch.Tensor, pad_value: Any) -> torch.Tensor:
    """Pad the last dimension of `tensor` on the right with a single value `pad_value`."""
    return F.pad(tensor, (0, 1), value=pad_value)

if __name__ == "__main__":
    input_tensor = torch.tensor([[1, 2], [3, 4]])
    pad_value = 5
    padded_tensor = pad_tensor_right(input_tensor, pad_value)
    print("Padded Tensor:", padded_tensor)
    print("Padding value used:", padded_tensor[-1, -1])