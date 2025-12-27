import torch
from x_transformers.x_transformers import XTransformer

def _create_causal_mask(decode_len: int, encode_len: int) -> torch.Tensor:
    """Create a lower-triangular mask of shape (decode_len, encode_len)."""
    return torch.tril(torch.ones(decode_len, encode_len))

def generate_shortened_sequence():
    """
    Instantiate an XTransformer model and generate a shortened sequence
    using a causal mask that maps a shorter decode length onto the original encoder length.
    """
    model = XTransformer()

    encoder_seq_len = 258
    decode_seq_len = 100

    # Example encoder input and decoder start tokens (kept as in original logic)
    src = torch.randn(encoder_seq_len, 512)
    start_tokens = torch.randn(32, model.config.nemb)

    mask = _create_causal_mask(decode_seq_len, encoder_seq_len)

    sample = model.generate(
        src,
        start_tokens,
        encoder_seq_len,
        mask=mask
    )

    return sample

generate_shortened_sequence()