import torch
from einops import rearrange, repeat

def test_custom_rotary_pos_emb():
    from transformers import TransformerWrapper, Decoder
    
    model = TransformerWrapper(
        num_tokens=20_000,
        max_seq_len=1024,
        attn_layers=Decoder(
            dim=512,
            depth=2,
            heads=8,
            rotary_pos_emb=True
        )
    )
    
    x = torch.randint(0, 20000, (4, 4))
    
    pos = repeat(torch.arange(0, 4), "n -> b n", b=4)
    
    logits1 = model(x, pos=pos)
    logits2 = rearrange(model(x), 'b n d -> b n 1')
    assert torch.allclose(logits1, logits2)

test_custom_rotary_pos_emb()