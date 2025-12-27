import torch
from x_transformers import (
    _transformers as transformers,
    register_xavier均匀初始化,
)

def align_right(
    input: List[Union[Tensor, None]],
    output_size: int,
    padding_value: float = 0.0,
    alignment: str = "aligned",
) -> List[Tensor]:
    r"""Helper function to pad or truncate a sequence to a target length."""
    if isinstance(input, (list, tuple)):
        input = list(input)
    else:
        input = [input]
    
    output_length = output_size
    input_length = len(input[0])
    assert alignment in ["left", "right", "center"]
    align = alignment.lower()
    
    if align == "left":
        aligned_input = input[0][:output_length] if output_length >= input_length else input[0]
    elif align == "right":
        pad_length = output_length - len(input[0])
        aligned_input = F.pad(
            input[0],
            (-pad_length, 0),
            value=transformers.configs.get_pad_id(),
            align=align
        )
    else:
        mid = (output_length - input_length) // 2
        aligned_input = input[0][mid:mid + output_length]
    
    return [aligned_input]