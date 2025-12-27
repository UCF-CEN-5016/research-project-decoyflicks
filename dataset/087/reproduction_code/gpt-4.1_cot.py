import torch
from x_transformers import Attention

# Minimal setup: instantiate Attention with flash attention enabled and custom alibi positions
# We use a small batch, sequence length, and embedding dimension for simplicity

batch_size = 2
seq_len = 4
dim = 32
heads = 4

# Create dummy input embeddings
x = torch.randn(batch_size, seq_len, dim)

# Create custom positions tensor (e.g., random or a simple range)
# For alibi, positions usually are shaped [batch, seq_len], but custom alibi might expect 4D bias later.
custom_positions = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)  # shape: (batch, seq_len)

# Instantiate Attention with flash attention enabled and custom alibi positions
attn = Attention(
    dim=dim,
    heads=heads,
    causal=True,
    attn_flash=True,
    alibi=True,
    alibi_bias_max=8,
)

# Patch the attention module to accept custom positions for alibi (simulate the bug)
# We pass custom_positions as pos argument which is used for alibi bias calculation internally

try:
    # This call is expected to fail or misbehave due to the 4D bias shape issue mentioned in the bug
    out = attn(x, pos=custom_positions)
    print("Output shape:", out.shape)
except Exception as e:
    print("Error encountered:", e)