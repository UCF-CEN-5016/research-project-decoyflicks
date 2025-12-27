import torch
import einops
from x_transformers import (
    AutoFlashAttention,
    PositionalEmbedding,
    TokenEmbedding,
    build attends,
)
from x_transformers.attend import FlashAttend

# Step 1: Set up minimal environment
device = "cpu"
dtype = torch.float32

# Step 2: Create a model with one layer and six positions to trigger the issue
class MinimalModel:
    def __init__(self, n_positions=6):
        self attends = build_attends(n_positions=n_positions)
        
    @torch.no_grad()
    def forward(self, x):
        # Custom alibi positions
        pe = PositionalEmbedding(d_model=128, device=device)
        custom_alibi_pos = torch.tensor([[i for i in range(6)]], dtype=torch.long).to(device)
        token_emb = TokenEmbedding(
            embed_dim=128,
            vocab_size=len(vocab),
            padding_idx=0
        )
        
        # Flash attention
        flash_attn = FlashAttend(attn_mask=True, attn_flash=True)
        
        return flash_attn(
            x,
            custom_alibi_pos,
            custom_alibi_pos,
            False  # mask
        )

# Step 3: Initialize the model and trigger the error
if __name__ == "__main__":
    vocab_size = 5000
    x = torch.randint(1, vocab_size, (1, 6), device=device)
    model = MinimalModel(n_positions=6)
    # This will reproduce the bug with flash attention when custom alibi positions are used