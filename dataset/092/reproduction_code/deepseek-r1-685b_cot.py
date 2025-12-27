import torch
from x_transformers import TransformerWrapper, Decoder

# Set up minimal decoder with cross-attention
decoder = TransformerWrapper(
    num_tokens=2049,
    max_seq_len=500,
    attn_layers=Decoder(
        dim=32,  # smaller dimension for minimal repro
        depth=1,  # single layer
        heads=2,  # minimal heads
        cross_attend=True,  # crucial for the bug
    )
)

# Create inputs
input_tokens = torch.randint(0, 2048, (2, 10))  # batch of 2, seq len 10
context = torch.rand(2, 5, 32)  # random context
context_mask = torch.zeros(2, 5, dtype=torch.bool)  # all padding (False)

# Forward pass - this will produce NaN
output = decoder(input_tokens, context=context, context_mask=context_mask)

# Verify NaN output
print("Output contains NaN:", torch.isnan(output).any().item())