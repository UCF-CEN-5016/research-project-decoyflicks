import torch
from x_transformers import TransformerWrapper, Decoder

torch.manual_seed(42)

num_tokens = 32
max_seq_len = 0
num_memory_tokens = 20

lm = TransformerWrapper(
    num_tokens=num_tokens,
    max_seq_len=max_seq_len,
    num_memory_tokens=num_memory_tokens,
    attn_layers=Decoder(
        dim=512,
        depth=4,
        heads=4,
        rotary_pos_emb=True,
        attn_flash=True,
        attn_onnxable=True,
        attn_num_mem_kv=20,
        attn_one_kv_head=True
    )
)

x = torch.randint(0, num_tokens, (8, 1024))

try:
    logits = lm(x)
except Exception as e:
    print(f"Error: {e}")

print(f"Input shape: {x.shape}, Input dtype: {x.dtype}")
print(f"Output shape: {logits.shape}, Output dtype: {logits.dtype}")