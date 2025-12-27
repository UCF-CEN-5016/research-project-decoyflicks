import torch
from einops import rearrange
from x_transformers import RotaryEmbedding

# Reproduction setup
dim = 32
seq_len = 1024
rotary_xpos = True
scale_base = 512

# Create rotary embedding instance
rotary_emb = RotaryEmbedding(
    dim=dim,
    custom=True,  # Using custom rotary embeddings
    rotary_xpos=rotary_xpos,
    scale_base=scale_base
)

# Simulate input that triggers the bug
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t = torch.randn(1, seq_len, dim).to(device)

# Positions tensor - this would normally come from the forward pass
positions = torch.arange(seq_len).to(device).unsqueeze(0)  # Shape [1, seq_len]

# This will trigger the bug when rotary_xpos=True
try:
    freqs, scale = rotary_emb(positions)
    print("Success! No bug triggered.")
except Exception as e:
    print(f"Bug triggered with error:\n{e}")