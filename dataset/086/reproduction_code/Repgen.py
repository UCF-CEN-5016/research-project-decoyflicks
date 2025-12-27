import torch
from x_transformers import TransformerWrapper, Decoder

def test_rotary_xpos_bug():
    # Create a model with both rotary_pos_emb and rotary_xpos
    model = TransformerWrapper(
        num_tokens=20_000,
        max_seq_len=1024,
        attn_layers=Decoder(
            dim=512,
            depth=2,
            heads=8,
            rotary_pos_emb=True,  # Enable the custom rotary embedding
            rotary_xpos=True      # This causes the conflict with the hack
        )
    )
    
    # Create random input tokens
    x = torch.randint(0, 20000, (1, 32))  # Match the shape in the error [1, 32]
    
    # Custom positions to trigger the hack
    pos = torch.arange(0, 32).unsqueeze(0)  # Shape [1, 32]
    
    # This should trigger the error
    try:
        output = model(x, pos=pos)
        print("Test passed without errors")
    except Exception as e:
        print(f"Error: {e}")
        print("Successfully reproduced the bug!")

test_rotary_xpos_bug()