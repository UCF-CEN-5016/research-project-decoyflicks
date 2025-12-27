import torch
from torch.nn.functional import gelu

batch_size = 100
sequence_length = 512

input_data = torch.rand((batch_size, sequence_length), dtype=torch.float32) * 2 - 1

try:
    gelu(input_data, approximate='True')
except TypeError as e:
    print(e)