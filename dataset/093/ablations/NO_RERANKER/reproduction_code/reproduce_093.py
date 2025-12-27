import torch
from x_transformers import TransformerWrapper, Decoder

torch.manual_seed(42)

batch_size = 8
input_dim = (8, 1024)
x = torch.randint(0, 32, input_dim)

model = TransformerWrapper(
    num_tokens=32,
    max_seq_len=0,
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

try:
    logits = model(x)
except Exception as e:
    print(f"Error: {e}")

print(f"x shape: {x.shape}, dtype: {x.dtype}")
print(f"logits shape: {logits.shape}, dtype: {logits.dtype}")