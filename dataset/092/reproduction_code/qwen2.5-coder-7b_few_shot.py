import torch
from x_transformers import TransformerWrapper, Decoder
from typing import Tuple


def build_decoder() -> TransformerWrapper:
    """
    Construct and return a TransformerWrapper configured as a decoder with cross-attention.
    """
    return TransformerWrapper(
        num_tokens=2049,
        max_seq_len=500,
        use_abs_pos_emb=True,
        scaled_sinu_pos_emb=True,
        attn_layers=Decoder(
            dim=1024,
            depth=24,
            heads=16,
            attn_dim_head=64,
            attn_flash=True,
            ff_no_bias=True,
            cross_attend=True,
        ),
    )


def prepare_inputs(
    batch_size: int = 2,
    seq_len: int = 20,
    vocab_upper_bound: int = 2048,
    context_len: int = 4,
    embed_dim: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create input token ids, context embeddings, and a context padding mask.
    The context_mask is all padding (all False -> padded).
    """
    input_tokens = torch.randint(0, vocab_upper_bound, (batch_size, seq_len))
    context_embeddings = torch.rand(batch_size, context_len, embed_dim)
    context_mask = torch.zeros(batch_size, context_len, dtype=torch.bool)
    return input_tokens, context_embeddings, context_mask


def run_decoder(
    decoder: TransformerWrapper,
    input_tokens: torch.Tensor,
    context_embeddings: torch.Tensor,
    context_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Run the decoder with the provided inputs and return the output.
    """
    return decoder(input_tokens, context=context_embeddings, context_mask=context_mask)


def main() -> None:
    decoder = build_decoder()
    input_tokens, context_embeddings, context_mask = prepare_inputs()
    output = run_decoder(decoder, input_tokens, context_embeddings, context_mask)
    print("Decoder output:", output)


if __name__ == "__main__":
    main()