import torch
from x_transformers import TransformerWrapper

# Minimal example to reproduce the bug and apply the fix
def test_cross_attention_all_padding():
    decoder = TransformerWrapper(
        num_tokens=2049,
        max_seq_len=500,
        use_abs_pos_emb=True,
        scaled_sinu_pos_emb=True,
        attn_layers=TransformerDecoderLayer(
            dim=1024,
            n_head=16,
            attn_dim_head=64,
            flash=True,
        ),
    )
    
    # Inputs
    i = torch.randint(0, 2048, (2, 20))
    context = torch.randn(2, 4, 1024)
    context_mask = torch.zeros(2, 4, dtype=torch.bool)  # All padding
    
    # Forward pass with fix
    out = decoder(i, context=context, context_mask=context_mask)
    
    # Check outputs for NaNs (should not occur after fix)
    assert not torch.isnan(out).any(), "Decoder output contains NaNs"
    print("Test passed: No NaNs in decoder output")

# Ensure the function is differentiable if gradients are needed
test_cross_attention_all_padding()