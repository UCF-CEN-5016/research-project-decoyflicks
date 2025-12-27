import torch
import torch.nn.functional as F

buggy_value = 'wrong_value'
def trigger_bug():
    try:
        output = F.gelu(torch.tensor([buggy_value]), approximate=buggy_value)
        print(output)
    except TypeError as e:
        print(f"Error: {e}")

trigger_bug()