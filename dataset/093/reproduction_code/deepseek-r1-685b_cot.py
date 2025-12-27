import torch
from x_transformers import TransformerWrapper, Decoder

# Minimal reproduction
lm = TransformerWrapper(
    num_tokens=32,
    max_seq_len=0,
    num_memory_tokens=20,
    attn_layers=Decoder(
        dim=512,
        depth=1,  # Minimal depth to reproduce
        heads=4,
        attn_num_mem_kv=20,  # Must be > 0
        attn_one_kv_head=True  # Must be True
    )
)

# Trigger forward pass
x = torch.randint(0, 32, (1, 64))  # Minimal batch and seq length
logits = lm(x)