import torch
from x_transformers.attend import Attend

torch.manual_seed(42)

batch_size = 2
sequence_length = 10
heads = 4
dim_head = 16

input_tensor = torch.randn(batch_size, sequence_length, heads * dim_head)
custom_position_tensor = torch.randint(0, sequence_length, (batch_size, sequence_length))

attention = Attend(
    dim=64,
    heads=heads,
    flash=True
)

output, intermediates = attention.forward(input_tensor, pos=custom_position_tensor)

assert output is not None
assert output.shape == (batch_size, sequence_length, heads * dim_head)
assert intermediates.qk_similarities.shape == (batch_size, heads, sequence_length, sequence_length)