import torch
from transformer_wrapper import TransformerWrapper, Decoder

# Initialize Transformer model
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
        attn_num_mem_kv=20,
        attn_one_kv_head=True
    )
)

# Generate random input tensor
batch_size = 8
seq_length = 1024
x = torch.randint(0, 32, (batch_size, seq_length))

# Forward pass through the model
logits = lm(x)

# Print tensor shapes and data types
print("Input shape:", x.shape, x.dtype)
print("Output shape:", logits.shape, logits.dtype)