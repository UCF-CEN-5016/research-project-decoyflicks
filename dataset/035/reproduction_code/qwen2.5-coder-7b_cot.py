from transformers import RotaryPositionalEmbeddings

ROTARY_DIM = 4

def create_rotary_positional_embeddings(dim: int = ROTARY_DIM) -> RotaryPositionalEmbeddings:
    """Factory for RotaryPositionalEmbeddings with a default dimension to address a reported issue."""
    return RotaryPositionalEmbeddings(dim)

rotary_positional_embeddings = create_rotary_positional_embeddings()