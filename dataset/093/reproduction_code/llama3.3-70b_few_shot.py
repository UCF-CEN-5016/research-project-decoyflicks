import torch
from transformers import TransformerWrapper, Decoder

# Incompatible configuration setup
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
        attn_num_mem_kv=20,  # Set to non-zero value
        attn_one_kv_head=True  # Set to True
    )
)

# Sample input data
x = torch.randint(0, 32, (8, 1024))

# This will cause an error
logits = lm(x)
print(x.shape, x.dtype)
print(logits.shape, logits.dtype)