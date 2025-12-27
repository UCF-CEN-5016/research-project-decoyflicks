import torch
from your_module import MultiInputTransformerWrapper, TransformerWrapper, Encoder, Decoder  # Adjust import based on your actual module structure

def main():
    # Step 1: Define parameters
    batch_size = 2
    seq_len = 1024

    # Step 2: Create input tensors
    x = dict(
        note=torch.randint(0, 20000, (batch_size, seq_len)),
        pitch=torch.randint(0, 32, (batch_size, seq_len)),
        tone=torch.randint(0, 16, (batch_size, seq_len))
    )

    # Step 3: Instantiate MultiInputTransformerWrapper
    model = MultiInputTransformerWrapper(
        num_tokens=dict(note=20000, pitch=32, tone=16),
        max_seq_len=seq_len,
        return_only_embed=True,
        attn_layers=Decoder(dim=128, depth=6, heads=8)
    )

    # Step 4: Run the model
    embed = model(x)
    assert embed.shape == (batch_size, seq_len, 128)

    # Step 5: Instantiate TransformerWrapper for average pooling
    model_avg_pool = TransformerWrapper(
        num_tokens=20000,
        max_seq_len=seq_len,
        num_memory_tokens=2,
        average_pool_embed=True,
        attn_layers=Encoder(dim=128, depth=6, heads=8)
    )

    # Step 6: Create input tensor for average pooling
    x_avg = torch.randint(0, 20000, (batch_size, seq_len))
    mask_avg = torch.randint(0, 2, (batch_size, seq_len)).bool()

    # Step 7: Run the model with average pooling
    logits_avg = model_avg_pool(x_avg, mask=mask_avg)
    assert logits_avg.shape == (batch_size, 20000)

    # Step 8: Instantiate TransformerWrapper for CLS token
    model_cls = TransformerWrapper(
        num_tokens=20000,
        max_seq_len=seq_len,
        num_memory_tokens=2,
        use_cls_token=True,
        attn_layers=Encoder(dim=128, depth=6, heads=8)
    )

    # Step 9: Create input tensor for CLS token
    x_cls = torch.randint(0, 20000, (batch_size, seq_len))
    mask_cls = torch.randint(0, 2, (batch_size, seq_len)).bool()

    # Step 10: Run the model with CLS token
    logits_cls = model_cls(x_cls, mask=mask_cls)
    assert logits_cls.shape == (batch_size, 20000)

    # Step 11: Instantiate TransformerWrapper for squeeze logit dimension
    model_squeeze = TransformerWrapper(
        num_tokens=20000,
        max_seq_len=seq_len,
        logits_dim=1,
        average_pool_embed=True,
        squeeze_out_last_dim=True,
        attn_layers=Encoder(dim=128, depth=6, heads=8)
    )

    # Step 12: Create input tensor for squeeze logit dimension
    x_squeeze = torch.randint(0, 20000, (batch_size, seq_len))
    mask_squeeze = torch.randint(0, 2, (batch_size, seq_len)).bool()

    # Step 13: Run the model with squeeze logit dimension
    logits_squeeze = model_squeeze(x_squeeze, mask=mask_squeeze)
    assert logits_squeeze.shape == (batch_size,)

    # Step 14: Instantiate TransformerWrapper for UNet skip
    model_unet = TransformerWrapper(
        num_tokens=20000,
        max_seq_len=seq_len,
        attn_layers=Encoder(dim=128, depth=6, heads=8, unet_skips=True)
    )

    # Step 15: Create input tensor for UNet skip
    x_unet = torch.randint(0, 20000, (batch_size, seq_len))
    mask_unet = torch.randint(0, 2, (batch_size, seq_len)).bool()

    # Step 16: Run the model with UNet skip
    model_unet(x_unet, mask=mask_unet)

    # Step 17: Check if encoder.to_logits.weight is part of the model parameters
    assert 'to_logits.weight' in dict(model_unet.named_parameters())

    # Step 18: Print the model parameters to verify if encoder.to_logits.weight is updating during training
    for name, param in model_unet.named_parameters():
        if 'to_logits.weight' in name:
            print(f"{name}: {param.data}")

if __name__ == "__main__":
    main()