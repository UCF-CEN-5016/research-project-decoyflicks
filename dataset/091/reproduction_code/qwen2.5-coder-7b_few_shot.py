import torch
from typing import Optional
from x_transformers import Transformer

# Configuration constants
_DIM = 64
_DEPTH = 2
_HEADS = 4
_FF_MULT = 4
_MAX_SEQ_LEN = 256
_NUM_TOKENS = 1000
_DEVICE = "cuda"

_START_TOKEN_ID = 1
_DECODE_SEQ_LEN = 200
_TEMPERATURE = 0.7
_BATCH_SIZE = 1


def build_transformer_model(device: str = _DEVICE) -> Transformer:
    """
    Construct and move the Transformer model to the specified device.
    """
    model = Transformer(
        dim=_DIM,
        depth=_DEPTH,
        heads=_HEADS,
        ff_mult=_FF_MULT,
        max_seq_len=_MAX_SEQ_LEN,
        num_tokens=_NUM_TOKENS
    ).to(device)
    return model


def generate_sequence(model: Transformer, input_tokens: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Generate a sequence from the given model and input tokens.
    Returns the generated output tensor, or None if a runtime error occurs.
    """
    try:
        with torch.no_grad():
            output = model.generate(
                input_tokens,
                start_tokens=torch.tensor([_START_TOKEN_ID]),  # Start token
                seq_len=_DECODE_SEQ_LEN,  # Decoding sequence length = 200 < 256
                temperature=_TEMPERATURE
            )
            return output
    except RuntimeError as exc:
        print("Error occurred:", str(exc))
        return None


def main() -> None:
    transformer = build_transformer_model()

    # Sample input sequence (longer than target output)
    input_seq = torch.randint(0, _NUM_TOKENS, (_BATCH_SIZE, _MAX_SEQ_LEN))  # Encoding sequence length = 256

    output = generate_sequence(transformer, input_seq)
    if output is not None:
        print("Generated output:", output.shape)


if __name__ == "__main__":
    main()