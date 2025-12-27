import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoregressiveWrapper(nn.Module):
    def __init__(self):
        super(AutoregressiveWrapper, self).__init__()

    def align_right(self, input_seq, pad_id=0):
        # This is where the bug occurs. The pad value is not being used.
        output = F.pad(input_seq, (1, 0), value=0)  # <--- Bug here
        return output

# Minimal environment setup
wrapper = AutoregressiveWrapper()

# Triggering conditions: Call align_right with a specific input
input_seq = torch.tensor([[1, 2], [3, 4]])
pad_id = 5  # This should be used in the F.pad call, but it's not

# Run the code to reproduce the bug
output = wrapper.align_right(input_seq, pad_id=pad_id)