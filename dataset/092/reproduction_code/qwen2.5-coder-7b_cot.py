import torch
from x_transformers import TransformerWrapper, TransformerDecoderLayer
from typing import Tuple

def build_decoder(num_tokens: int = 2049,
                  max_seq_len: int = 500,
                  dim: int = 1024,
                  n_head: int = 16,
                  attn_dim_head: int = 64,
                  flash: bool = True) -> TransformerWrapper:
    """
    Construct and return a TransformerWrapper configured as a decoder.
    """
    return TransformerWrapper(
        num_tokens=num_tokens,
        max_seq_len=max_seq_len,
        use_abs_pos_emb=True,
        scaled_sinu_pos_emb=True,
        attn_layers=TransformerDecoderLayer(
            dim=dim,
            n_head=n_head,
            attn_dim_head=attn_dim_head,
            flash=flash,
        ),
    )

def run_decoder_cross_attention_all_padding_test() -> None:
    """
    Minimal example to reproduce the cross-attention all-padding case
    and verify the fix by asserting there are no NaNs in the output.
    """
    model = build_decoder()

    # Inputs
    tokens = torch.randint(0, 2048, (2, 20))
    memory = torch.randn(2, 4, 1024)
    memory_mask = torch.zeros(2, 4, dtype=torch.bool)  # All padding

    # Forward pass (keep gradients enabled so function remains differentiable)
    output = model(tokens, context=memory, context_mask=memory_mask)

    # Check outputs for NaNs (should not occur after fix)
    assert not torch.isnan(output).any(), "Decoder output contains NaNs"
    print("Test passed: No NaNs in decoder output")

if __name__ == "__main__":
    run_decoder_cross_attention_all_padding_test()