import torch
from transformer_wrapper import TransformerWrapper

# Create a Transformer model with incorrect configuration
lm = TransformerWrapper(
    num_tokens=32,
    max_seq_len=0,
    num_memory_tokens=20,
    attn_layers=TransformerWrapper.Decoder(
        dim=512,
        depth=4,
        heads=4,
        rotary_pos_emb=True,
        attn_flash=True,
        attn_onnxable=True,
        attn_num_mem_kv=20,
        attn_one_kv_head=True  # This causes the error
    )
)

# Test input
x = torch.randint(0, 32, (8, 1024))

try:
    logits = lm(x)
    print(x.shape, x.dtype)
    print(logits.shape, logits.dtype)
except Exception as e:
    print("Error:", str(e))