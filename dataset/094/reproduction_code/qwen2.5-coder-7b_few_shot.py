import torch
from typing import Optional, List, Tuple, Any

class ContinuousTransformerWrapper(torch.nn.Module):
    """
    Lightweight wrapper simulating a transformer with external memory.
    Core behavior:
      - Stores configuration passed at init
      - forward returns the input tensor unchanged and optionally updated memories
    """
    def __init__(
        self,
        dim_out: int,
        max_seq_len: int,
        num_memory_tokens: int,
        max_mem_len: int,
        attn_layers: Any
    ) -> None:
        super().__init__()
        self.dim_out = dim_out
        self.max_seq_len = max_seq_len
        self.num_memory_tokens = num_memory_tokens
        self.max_mem_len = max_mem_len
        self.attn_layers = attn_layers

    def update_mems(self, inputs: torch.Tensor, memories: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Placeholder for memory-update logic. Returns memories unchanged.
        """
        return memories

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor],
        mems: Optional[List[torch.Tensor]],
        return_mems: bool
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Simulated forward pass. If mems provided, call update_mems and return them.
        """
        updated_mems = mems
        if mems is not None:
            updated_mems = self.update_mems(inputs, mems)
        return inputs, updated_mems


if __name__ == "__main__":
    # Build a simple configuration for the wrapper.
    # Note: using plain dict for attn_layers configuration (not actual modules).
    attn_config = {
        'Decoder': {
            'dim': 512,
            'depth': 6,
            'heads': 4,
            'rotary_pos_emb': True,
            'shift_tokens': 1,
            'attn_flash': True,
            'attn_onnxable': True,
            'use_rmsnorm': True,
            'sandwich_norm': True
        }
    }

    net = ContinuousTransformerWrapper(
        dim_out=259,
        max_seq_len=0,
        num_memory_tokens=20,
        max_mem_len=100,
        attn_layers=attn_config
    )

    inputs = torch.randn(1, 1024, 512)
    mask_tensor = torch.randn(1, 1024) > 0
    memories = [torch.zeros(1, 100, 512) for _ in range(6)]

    logits, returned_mems = net(inputs, mask=mask_tensor, mems=memories, return_mems=True)

    print("Logits shape:", logits.shape)
    print("Mems shapes:", [m.shape for m in returned_mems])