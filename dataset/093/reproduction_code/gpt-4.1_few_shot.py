import torch
from some_module import TransformerWrapper, Decoder  # Replace with actual import paths

# Attempt to create a model with conflicting attention settings
lm = TransformerWrapper(
    num_tokens=32,
    max_seq_len=0,
    num_memory_tokens=20,
    attn_layers=Decoder(
        dim=512,
        depth=4,
        heads=4,
        rotary_pos_emb=True,
        attn_flash=True,
        attn_onnxable=True,
        attn_num_mem_kv=20,      # > 0
        attn_one_kv_head=True    # True at the same time triggers error
    )
)

# Input batch of token indices
x = torch.randint(0, 32, (8, 1024))

# This call raises an error due to conflicting configs
logits = lm(x)

print(x.shape, x.dtype)
print(logits.shape, logits.dtype)