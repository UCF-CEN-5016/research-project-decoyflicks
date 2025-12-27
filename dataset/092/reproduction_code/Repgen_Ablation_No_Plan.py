import torch
from x_transformers import TransformerWrapper, Decoder

def reproduce_bug():
    model = TransformerWrapper(
        num_tokens=20000,
        max_seq_len=1024,
        attn_layers=Decoder(
            dim=512,
            dim_condition=768,
            depth=12,
            heads=8
        )
    )

    i = torch.randint(0, 256, (2, 1024))
    context = torch.randn(2, 768)
    context_mask = torch.zeros_like(context).bool()

    logits = model(i, context=context)
    print(logits)

reproduce_bug()