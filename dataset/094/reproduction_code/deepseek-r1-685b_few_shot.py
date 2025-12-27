import torch
from einops import rearrange
from x_transformers import ContinuousTransformerWrapper, Decoder

# Initialize model with memory settings
model = ContinuousTransformerWrapper(
    dim_out=256,
    max_seq_len=0,
    num_memory_tokens=20,
    max_mem_len=100,
    attn_layers=Decoder(
        dim=512,
        depth=6,
        heads=8,
        rotary_pos_emb=True
    )
)

# Create test inputs
x = torch.randn(1, 1024, 512)  # (batch, seq, dim)
mask = torch.randn(1, 1024) > 0  # Random mask
mems = [torch.zeros(1, 100, 512) for _ in range(6)]  # Initial memory

# This fails despite matching the expected memory structure
try:
    logits, new_mems = model(x, mask=mask, mems=mems, return_mems=True)
    print("Success!")
    print("Logits shape:", logits.shape)
    print("Memory shapes:", [m.shape for m in new_mems])
except Exception as e:
    print("Failed with error:", e)

# This works (mems=None)
logits, new_mems = model(x, mask=mask, mems=None, return_mems=True)
print("Works with mems=None")
print("Logits shape:", logits.shape)
print("Memory shapes:", [m.shape for m in new_mems])