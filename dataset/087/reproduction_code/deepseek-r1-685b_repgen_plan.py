import torch
from x_transformers import TransformerWrapper, Decoder

# Model setup with flash attention
model = TransformerWrapper(
    num_tokens=256,
    max_seq_len=512,
    attn_layers=Decoder(
        dim=512,
        depth=6,
        heads=8,
        attn_flash=True,  # Enable flash attention
        attn_alibi=True   # Enable alibi position bias
    )
).cuda()

# Custom 4D attention bias (batch, heads, seq, seq)
batch_size = 2
seq_len = 128
custom_alibi = torch.randn(batch_size, 8, seq_len, seq_len).cuda()

# Input data
x = torch.randint(0, 256, (batch_size, seq_len)).cuda()

# Handle 4D alibi bias for flash attention
if model.attn_layers.attn_flash and model.attn_layers.attn_alibi:
    # Reshape the custom alibi bias to match the model's expected shape
    custom_alibi = custom_alibi.unsqueeze(1).repeat(1, model.attn_layers.heads, 1, 1)
    out = model(x, attn_bias=custom_alibi)
    print("Forward pass successful")
else:
    print("Flash attention failed to handle 4D alibi bias")