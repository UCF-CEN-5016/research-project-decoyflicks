import torch
from your_module import MultiInputTransformerWrapper, TransformerWrapper, Decoder, Encoder  # Replace 'your_module' with the actual module name

def main():
    batch_size = 2
    seq_len = 1024

    x = {
        'note': torch.randint(0, 20000, (batch_size, seq_len)),
        'pitch': torch.randint(0, 32, (batch_size, seq_len)),
        'tone': torch.randint(0, 16, (batch_size, seq_len))
    }

    model = MultiInputTransformerWrapper(
        num_tokens={'note': 20000, 'pitch': 32, 'tone': 16},
        max_seq_len=seq_len,
        return_only_embed=True,
        attn_layers=Decoder(dim=128, depth=6, heads=8)
    )

    embed = model(x)
    assert embed.shape == (batch_size, seq_len, 128)

    model_logits = TransformerWrapper(
        num_tokens=20000,
        max_seq_len=seq_len,
        num_memory_tokens=2,
        average_pool_embed=True,
        attn_layers=Encoder(dim=128, depth=6, heads=8)
    )

    x_input = torch.randint(0, 20000, (batch_size, seq_len))
    mask = torch.randint(0, 2, (batch_size, seq_len)).bool()

    logits = model_logits(x_input, mask=mask)
    assert logits.shape == (batch_size, 20000)

    model_cls = TransformerWrapper(
        num_tokens=20000,
        max_seq_len=seq_len,
        num_memory_tokens=2,
        use_cls_token=True,
        attn_layers=Encoder(dim=128, depth=6, heads=8)
    )

    logits_cls = model_cls(x_input, mask=mask)
    assert logits_cls.shape == (batch_size, 20000)

    model_squeeze = TransformerWrapper(
        num_tokens=20000,
        max_seq_len=seq_len,
        logits_dim=1,
        average_pool_embed=True,
        squeeze_out_last_dim=True,
        attn_layers=Encoder(dim=128, depth=6, heads=8)
    )

    logits_squeeze = model_squeeze(x_input, mask=mask)
    assert logits_squeeze.shape == (batch_size,)

    weight_before = model_logits.encoder.to_logits.weight.data.clone()
    logits = model_logits(x_input, mask=mask)
    weight_after = model_logits.encoder.to_logits.weight.data

    print("Weight before:", weight_before)
    print("Weight after:", weight_after)

if __name__ == "__main__":
    main()