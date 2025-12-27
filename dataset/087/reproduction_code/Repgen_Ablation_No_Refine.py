import torch
from x_transformers import XTransformer

def reproduce_bug_with_xtransformer():
    # Create a model with alibi positional bias and flash attention using XTransformer
    model = XTransformer(
        dim=512,
        enc_num_tokens=20000,
        enc_depth=6,
        enc_heads=8,
        enc_max_seq_len=512,
        dec_num_tokens=20000,
        dec_depth=6,
        dec_heads=8,
        dec_max_seq_len=512,
        tie_token_emb=True,
        alibi_pos_bias=True,  # Enable alibi positional bias
        attn_flash=True       # Enable flash attention
    )
    
    # Create input tokens
    src = torch.randint(0, 20000, (2, 256))
    tgt = torch.randint(0, 20000, (2, 128))
    
    # Create custom alibi positions
    enc_alibi_pos = torch.arange(256).unsqueeze(0).repeat(2, 1)
    dec_alibi_pos = torch.arange(128).unsqueeze(0).repeat(2, 1)
    
    try:
        # This should fail when using custom positions with flash attention
        output = model(
            src, 
            tgt, 
            enc_alibi_pos=enc_alibi_pos,
            dec_alibi_pos=dec_alibi_pos
        )
        print("✓ Test unexpectedly passed")
        return output
    except Exception as e:
        print(f"✗ Bug reproduced: {e}")
        return None

output = reproduce_bug_with_xtransformer()