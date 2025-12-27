import torch
from x_transformers import TransformerWrapper, Decoder

# Initialize decoder with cross-attention
decoder = TransformerWrapper(
    num_tokens=2049,
    max_seq_len=500,
    attn_layers=Decoder(
        dim=1024,
        depth=2,  # Reduced for minimal repro
        heads=16,
        cross_attend=True  # Enable cross-attention
    )
)

# Sample inputs
tokens = torch.randint(0, 2048, (2, 20))  # Random tokens
context = torch.rand(2, 4, 1024)  # Random context
context_mask = torch.zeros(2, 4, dtype=torch.bool)  # All padding

# Forward pass produces NaN
output = decoder(tokens, context=context, context_mask=context_mask)
print("Output contains NaN:", torch.isnan(output).any())