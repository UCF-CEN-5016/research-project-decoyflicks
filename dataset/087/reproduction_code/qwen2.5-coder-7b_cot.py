import torch
import einops
from x_transformers import (
    AutoFlashAttention,
    PositionalEmbedding,
    TokenEmbedding,
    build_attends,
)
from x_transformers.attend import FlashAttend

# Environment configuration
DEVICE = "cpu"
DTYPE = torch.float32

class MinimalFlashModel:
    def __init__(self, n_positions: int = 6):
        self.n_positions = n_positions
        # preserve original behavior of building attends
        self.attends = build_attends(n_positions=n_positions)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # positional embedding (kept for parity with original)
        pe = PositionalEmbedding(d_model=128, device=DEVICE)

        # custom alibi positions matching original construction
        custom_alibi_pos = torch.arange(self.n_positions, device=DEVICE).unsqueeze(0).long()

        # token embedding (constructed but not used further, matching original)
        token_emb = TokenEmbedding(
            embed_dim=128,
            vocab_size=VOCAB_SIZE,
            padding_idx=0
        )

        # use FlashAttend with same flags as original
        flash_attn = FlashAttend(attn_mask=True, attn_flash=True)

        # call FlashAttend preserving original argument order and mask flag
        return flash_attn(
            x,
            custom_alibi_pos,
            custom_alibi_pos,
            False  # mask
        )

if __name__ == "__main__":
    VOCAB_SIZE = 5000
    x = torch.randint(1, VOCAB_SIZE, (1, 6), device=DEVICE)
    model = MinimalFlashModel(n_positions=6)
    # This will reproduce the bug with flash attention when custom alibi positions are used
    _ = model.forward(x)