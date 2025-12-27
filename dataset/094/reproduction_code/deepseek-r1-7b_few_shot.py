import torch
from .ContinuousTransformerWrapper import ContinuousTransformerWrapper

model = ContinuousTransformerWrapper(
    dim_out=259,
    max_seq_len=0,
    num_memory_tokens=20,
    max_mem_len=100,
    # Other parameters remain unchanged
)

x = torch.randn(1, 1024, 512)  # Batch size 1, sequence length 1024, features 512
m = torch.ones(1, 1024) > 0     # Mask indicating which positions are valid

# Initialize memory with an invalid structure (e.g., fewer tokens than allowed)
mems = [torch.zeros(1, 100, 512) for _ in range(6)]  # Attempts to return only 6 memories

try:
    output = model(x, mask=m, return_mems=True, mems=mems)
except Exception as e:
    print(f"An error occurred when using return_mems=True with an invalid mems list: {e}")