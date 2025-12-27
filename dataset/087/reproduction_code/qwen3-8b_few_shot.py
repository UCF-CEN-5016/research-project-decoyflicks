import torch
from x_transformers import Attend  # Assuming the Attend class is available

# Custom alibi position function that returns 4D tensor
def custom_alibi_pos(pos):
    return torch.randn(2, 3, 4, 5)  # Example 4D shape

# Create sample input tensors
query = torch.randn(2, 3, 4, 5)
key = torch.randn(2, 3, 4, 5)

# This will fail with shape mismatch when using flash attention
attention = Attend(
    dim=5,
    alibi_pos=custom_alibi_pos,
    attn_flash=True
)
output = attention(query, key)