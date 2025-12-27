import torch
from x_transformers import TransformerWrapper, Decoder

def test_custom_alibi_without_flash():
    # Create a model with alibi positional bias but WITHOUT flash attention
    model = TransformerWrapper(
        num_tokens=20000,
        max_seq_len=1024,
        attn_layers=Decoder(
            dim=512,
            depth=6,
            heads=8,
            alibi_pos_bias=True,  # Enable alibi positional bias
            attn_flash=False      # Disable flash attention
        )
    )
    
    # Create input tokens
    x = torch.randint(0, 20000, (2, 512))
    
    # Create custom alibi positions
    alibi_pos = torch.arange(512).unsqueeze(0).repeat(2, 1)
    
    try:
        # This should work fine without flash attention
        output = model(x, alibi_pos=alibi_pos)
        print("✓ Test passed as expected (no flash attention)")
        return output
    except Exception as e:
        print(f"✗ Test unexpectedly failed: {e}")
        return None

output = test_custom_alibi_without_flash()