import torch
from x_transformers import TransformerWrapper, Decoder

def initialize_decoder():
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
    return decoder

def sample_inputs():
    # Sample inputs
    tokens = torch.randint(0, 2048, (2, 20))  # Random tokens
    context = torch.rand(2, 4, 1024)  # Random context
    context_mask = torch.zeros(2, 4, dtype=torch.bool)  # All padding
    return tokens, context, context_mask

def check_for_nan(output):
    # Check if output contains NaN
    contains_nan = torch.isnan(output).any()
    print("Output contains NaN:", contains_nan)

decoder = initialize_decoder()
tokens, context, context_mask = sample_inputs()
output = decoder(tokens, context=context, context_mask=context_mask)
check_for_nan(output)