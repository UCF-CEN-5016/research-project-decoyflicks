import torch
import xtransformers


def build_decoder_wrapper() -> xtransformers.TransformerWrapper:
    """
    Construct and return a TransformerWrapper configured as a decoder with the same
    architecture parameters as the original code.
    """
    return xtransformers.TransformerWrapper(
        num_tokens=2049,
        max_seq_len=500,
        use_abs_pos_emb=True,
        scaled_sinu_pos_emb=True,
        attn_layers=xtransformers.Decoder(
            dim=1024,
            depth=24,
            heads=16,
            attn_dim_head=64,
            attn_flash=True,
            ff_no_bias=True,
            cross_attend=True,
        ),
    )


def prepare_dummy_inputs(
    batch_size: int = 2,
    seq_len: int = 20,
    context_len: int = 4,
    embed_dim: int = 1024,
    randint_upper: int = 2048,
):
    """
    Prepare token ids, context embeddings, and a context mask matching the shapes
    used by the original example.
    """
    token_ids = torch.randint(0, randint_upper, (batch_size, seq_len))
    context_embeddings = torch.rand(batch_size, context_len, embed_dim)
    context_mask = torch.zeros(batch_size, context_len, dtype=torch.bool)
    return token_ids, context_embeddings, context_mask


def run_example():
    decoder_model = build_decoder_wrapper()
    token_ids, context_embeddings, context_mask = prepare_dummy_inputs()
    output = decoder_model(token_ids, context=context_embeddings, context_mask=context_mask)
    print(output)


if __name__ == "__main__":
    run_example()