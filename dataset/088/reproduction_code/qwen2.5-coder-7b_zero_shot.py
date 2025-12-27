import torch
import torch.nn as nn

class RightAlignPadder(nn.Module):
    """Pad each element of a 1D tensor to the right with constant values and flatten."""
    def __init__(self, left_pad: int = 0, right_pad: int = 2, fill_value: int = 0):
        super().__init__()
        self._padder = nn.ConstantPad1d((left_pad, right_pad), fill_value)

    def align_right(self, tensor: torch.Tensor) -> torch.Tensor:
        column = tensor.view(-1, 1)
        padded = self._padder(column)
        return padded.view(-1)

if __name__ == "__main__":
    pad_model = RightAlignPadder()
    print(pad_model.align_right(torch.tensor([1, 2, 3, 4, 5])))