import torch
from x_transformers import TransformerWrapper, Decoder

# Initialize the decoder model
decoder = TransformerWrapper(
    num_tokens=2049,
    max_seq_len=500,
    use_abs_pos_emb=True,
    scaled_sinu_pos_emb=True,
    attn_layers=Decoder(
        dim=1024,
        depth=24,
        heads=16,
        attn_dim_head=64,
        attn_flash=True,
        ff_no_bias=True,
        cross_attend=True,
    ),
)

# Generate a random input sequence
input_seq = torch.randint(0, 2048, (2, 20))

# Create a context with all padding (context_mask is all False)
context = torch.rand(2, 4, 1024)
context_mask = torch.zeros(2, 4, dtype=torch.bool)

# Pass the input sequence and context through the decoder
output = decoder(input_seq, context=context, context_mask=context_mask)

# Print the output (expected to be all NaN)
print("Decoder Output:", output)