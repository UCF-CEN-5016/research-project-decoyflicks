import torch
from x_transformers import TransformerWrapper, Decoder

def reproduce_alibi_flash_bug():
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
    
    # Create custom alibi positions
    alibi_pos = torch.arange(512).unsqueeze(0).repeat(2, 1)
    
    try:
        # This should fail when using custom positions with flash attention
        output = model(x, alibi_pos=alibi_pos)
        print("✓ Test unexpectedly passed")
        return output
    except Exception as e:
        print(f"✗ Bug reproduced: {e}")
        return None

output = reproduce_alibi_flash_bug()