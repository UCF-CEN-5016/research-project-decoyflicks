import torch
from x_transformers import XTransformer

# Minimal setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Custom ALiBi positional bias tensor (4D) - for batch and heads
# Shape: (batch_size, num_heads, seq_len, seq_len)
batch_size, num_heads, seq_len = 2, 4, 8
custom_alibi = torch.randn(batch_size, num_heads, seq_len, seq_len, device=device)

# Model with flash attention enabled and custom alibi passed
model = XTransformer(
    dim=32,
    depth=1,
    heads=num_heads,
    attn_flash=True,  # Enable flash attention
    alibi=True
).to(device)

x = torch.randn(batch_size, seq_len, 32, device=device)

try:
    # Pass the custom alibi bias directly (simulate custom position alibi)
    output = model(x, alibi_pos=custom_alibi)
    print("Output:", output.shape)
except Exception as e:
    print("Error encountered:", e)