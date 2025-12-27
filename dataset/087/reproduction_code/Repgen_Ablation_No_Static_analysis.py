import torch
from x_transformers import TransformerWrapper, Decoder

def test_flash_without_custom_alibi():
    # Create a model with alibi positional bias and flash attention
    model = TransformerWrapper(
        num_tokens=20000,
        max_seq_len=1024,
        attn_layers=Decoder(
            dim=512,
            depth=6,
            heads=8,
            alibi_pos_bias=True,  # Enable alibi positional bias
            attn_flash=True       # Enable flash attention
        )
    )
    
    # Create input tokens
    x = torch.randint(0, 20000, (2, 512))
    
    # Don't provide custom alibi positions
    try:
        # This should work with flash attention but no custom positions
        output = model(x)  # No alibi_pos parameter
        print("✓ Test passed as expected (flash but no custom positions)")
        return output
    except Exception as e:
        print(f"✗ Test unexpectedly failed: {e}")
        return None

output = test_flash_without_custom_alibi()