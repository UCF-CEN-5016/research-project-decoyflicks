import torch
from torch import nn
from x_transformers import TransformerWrapper, Decoder

def create_transformer_model():
    num_tokens = 256
    max_seq_len = 512
    dim = 512
    depth = 6
    heads = 8
    kv_heads = 4
    qk_norm = kv_heads != heads
    dim_head = dim // heads

    transformer_config = Decoder(
        dim=dim,
        depth=depth,
        heads=heads,
        kv_heads=kv_heads,
        qk_norm=qk_norm,
        dim_head=dim_head
    )

    model = TransformerWrapper(
        num_tokens=num_tokens,
        max_seq_len=max_seq_len,
        attn_layers=transformer_config
    )

    return model

def main():
    model = create_transformer_model()

    # Forward pass with sample input
    x = torch.randint(0, 256, (1, 128))
    out = model(x)  # Will raise shape mismatch error

    print(out.shape)

if __name__ == "__main__":
    main()