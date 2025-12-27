import torch
from your_module import TransformerWrapper, Decoder  # Replace 'your_module' with the actual module name

# Set up minimal environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Add triggering conditions
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
        attn_num_mem_kv=20,  # Triggering condition 1
        attn_one_kv_head=True  # Triggering condition 2
    )
)

# Create input tensor
x = torch.randint(0, 32, (8, 1024), device=device)

# Attempt to reproduce the bug
try:
    logits = lm(x)
    print(x.shape, x.dtype)
    print(logits.shape, logits.dtype)
except Exception as e:
    print(f"Error: {e}")