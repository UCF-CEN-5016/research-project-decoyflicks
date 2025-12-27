import torch
import math
from einops import rearrange, repeat
from x_transformers import TransformerWrapper, Decoder

class CustomRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, pos=None):
        seq_len = x.shape[1]
        if pos is None:
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        else:
            t = pos.type_as(self.inv_freq)
        
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.sin(), emb.cos()

def test_rotary_with_xpos():
    # Set up the model with both rotary features
    model = TransformerWrapper(
        num_tokens=20_000,
        max_seq_len=1024,
        attn_layers=Decoder(
            dim=512,
            depth=2,
            heads=8,
            rotary_pos_emb=True,
            rotary_xpos=True,
            custom_rotary_emb=CustomRotaryEmbedding(dim=64)  # This is part of the hack
        )
    )
    
    # Input with shape that will trigger the error
    x = torch.randint(0, 20000, (1, 32))
    
    # Try with and without custom positions
    try:
        # Without custom positions (should work)
        output1 = model(x)
        print("Test without custom positions passed")
        
        # With custom positions (should fail due to shape mismatch)
        pos = torch.arange(0, 32).unsqueeze(0)  # Shape [1, 32]
        output2 = model(x, pos=pos)
        print("Test with custom positions passed (unexpected)")
    except Exception as e:
        print(f"Error: {e}")
        print("Bug reproduced as expected")

test_rotary_with_xpos()