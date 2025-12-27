import torch
import torch.nn.functional as F

def align_right(tensor, pad_id, max_len):
    # Intended to pad tensor on the left with pad_id, but pads with 0 instead
    # Bug: value=0 is hardcoded instead of using pad_id
    pad_len = max_len - tensor.size(-1)
    padded = F.pad(tensor, (pad_len, 0), value=0)  # Should be value=pad_id
    return padded

# Example usage
pad_id = 9
x = torch.tensor([1, 2, 3])
max_len = 5

result = align_right(x, pad_id, max_len)
print("Padded tensor:", result)
print("Expected left padding with pad_id=9, but got zeros instead.")