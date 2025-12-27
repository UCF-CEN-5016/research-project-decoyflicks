import torch
from x_transformers import TransformerWrapper, Encoder, Decoder

torch.manual_seed(42)

batch_size = 8
input_dim = (8, 1024)
x = torch.randint(0, 32, input_dim)

attn_layers = Decoder(dim=512, depth=4, heads=4, rotary_pos_emb=True, attn_flash=True, attn_onnxable=True, attn_num_mem_kv=20, attn_one_kv_head=True)

lm = TransformerWrapper(num_tokens=32, max_seq_len=0, attn_layers=attn_layers, num_memory_tokens=20)

logits = lm(x)

print(f"Input shape: {x.shape}, dtype: {x.dtype}")
print(f"Output shape: {logits.shape}, dtype: {logits.dtype}")
assert not torch.isnan(logits).any()