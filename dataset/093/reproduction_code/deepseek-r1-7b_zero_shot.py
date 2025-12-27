import torch
from torch.nn import TransformerWrapper
from transformers import Decoder

class TransformerWrapper(TransformerWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if kwargs.get('attn_num_mem_kv') > 0 and kwargs.get('attn_one_kv_head'):
            raise ValueError("Cannot set both attn_num_mem_kv > 0 and attn_one_kv_head. They conflict.")

# Create the model with fixed parameters
lm = TransformerWrapper(
    num_tokens          = 32,
    max_seq_len         = 0,
    num_memory_tokens   = 20,
    attn_layers = Decoder (
        dim             = 512,
        depth           = 4,
        heads           = 4,
        rotary_pos_emb  = True,
        attn_flash      = True,
        attn_onnxable   = True,
        # Only set num_mem_kv=20 and prevent using one_kv_head
        attn_num_mem_kv  = 20,
        attn_one_kv_head = False  # This prevents the conflict
    )
)

x = torch.randint(0, 32, (8, 1024))
logits = lm(x)
print(f"x shape: {x.shape}, dtype: {x.dtype}")
print(f"logits shape: {logits.shape}, dtype: {logits.dtype}")