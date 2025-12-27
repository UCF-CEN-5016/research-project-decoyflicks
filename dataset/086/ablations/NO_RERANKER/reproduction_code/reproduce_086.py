import torch
from einops import rearrange
from x_transformers import RotaryEmbedding, apply_rotary_pos_emb
import einops  # Importing einops to fix the undefined variable issue

torch.manual_seed(42)

batch_size = 1
sequence_length = 32
embedding_dim = 64

# Generate random input tensor and frequency tensor
x = torch.rand(batch_size, sequence_length, embedding_dim)
freqs = torch.rand(batch_size, sequence_length, embedding_dim // 2)

# Initialize rotary embedding with xpos enabled
rotary_embedding = RotaryEmbedding(dim=64, use_xpos=True, scale_base=512, interpolation_factor=1.0)
freqs, scale = rotary_embedding(x)

# Scale tensor for rotary position embedding
scale = torch.rand(batch_size, sequence_length, 1)

# Attempt to apply rotary position embedding and catch specific errors
try:
    output = apply_rotary_pos_emb(x, freqs, scale)
except Exception as e:
    # Check if the error is related to einops and print the bug reproduction message
    if isinstance(e, einops.EinopsError) and 'Error while processing rearrange-reduction pattern "n -> n 1"' in str(e):
        print("Bug reproduced:", str(e))