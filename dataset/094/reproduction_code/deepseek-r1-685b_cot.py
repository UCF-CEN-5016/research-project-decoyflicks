import torch
from x_transformers import ContinuousTransformerWrapper, Decoder

# Minimal model setup
net = ContinuousTransformerWrapper(
    dim_out = 16,  # reduced for minimal repro
    max_seq_len = 0,
    num_memory_tokens = 4,  # reduced from 20
    max_mem_len = 10,  # reduced from 100
    attn_layers = Decoder(
        dim = 32,  # reduced from 512
        depth = 2,  # reduced from 6
        heads = 2,  # reduced from 4
        rotary_pos_emb = True
    )
)

# Input tensors (reduced dimensions)
x = torch.randn(1, 16, 32)  # (batch, seq, dim)
mask = torch.randn(1, 16) > 0

# This works fine
logits, mems = net(x, mask=mask, mems=None, return_mems=True)
print("Works with mems=None")
print(logits.shape)
print([m.shape for m in mems])

# This triggers the bug
mems = [torch.zeros(1, 10, 32) for _ in range(2)]  # matching depth
print("\nFails with pre-initialized mems:")
try:
    logits, mems = net(x, mask=mask, mems=mems, return_mems=True)
    print(logits.shape)
    print([m.shape for m in mems])
except Exception as e:
    print(f"Error: {e}")