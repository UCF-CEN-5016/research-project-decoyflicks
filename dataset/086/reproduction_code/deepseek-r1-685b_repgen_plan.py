import torch
from x_transformers import RotaryEmbedding

# Setup rotary embedding with xpos enabled
rotary_emb = RotaryEmbedding(
    dim=32,  # Example dimension
    xpos_scale_base=512,  # Enable xpos
    custom=True  # Trigger the problematic path
)

# Simulate input tensors (batch_size=1, seq_len=32, dim=32)
q = torch.randn(1, 32, 32)
k = torch.randn(1, 32, 32)

# This will trigger the shape mismatch error
try:
    q_rotated, k_rotated = rotary_emb.rotate_queries_and_keys(q, k)
except Exception as e:
    print(f"Error occurred: {e}")
    print("The issue happens when trying to rearrange a 2D tensor as 1D in xpos calculation")