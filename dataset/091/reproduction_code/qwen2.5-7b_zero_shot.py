import torch
from x_transformers import Transformer

def build_model(dim=64, depth=2, heads=4, encoder_depth=2, encoder_heads=4, decoder_depth=2, decoder_heads=4, num_tokens=1000, max_seq_len=256):
    model = Transformer(
        dim=dim,
        depth=depth,
        heads=heads,
        encoder_depth=encoder_depth,
        encoder_heads=encoder_heads,
        decoder_depth=decoder_depth,
        decoder_heads=decoder_heads,
        num_tokens=num_tokens,
        max_seq_len=max_seq_len
    )
    return model

def main():
    model = build_model()

    src = torch.randint(0, 1000, (1, 256))
    src_mask = torch.ones(1, 1, 256, 256)

    sample = model.generate(
        src, 
        start_tokens=torch.tensor([1]), 
        seq_len=200, 
        mask=src_mask
    )

if __name__ == "__main__":
    main()