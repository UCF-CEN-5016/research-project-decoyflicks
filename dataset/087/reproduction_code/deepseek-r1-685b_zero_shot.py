import torch
from x_transformers import TransformerWrapper, Decoder

def main():
    model = TransformerWrapper(
        num_tokens=256,
        max_seq_len=512,
        attn_layers=Decoder(
            dim=64,
            depth=1,
            heads=8,
            attn_flash=True,
            alibi_pos_bias=True,
            alibi_pos_bias_dim_scale=2
        )
    )

    x = torch.randint(0, 256, (1, 10))
    pos = torch.rand(1, 10, 8) * 2 - 1  # Custom positions
    out = model(x, pos=pos)

if __name__ == '__main__':
    main()