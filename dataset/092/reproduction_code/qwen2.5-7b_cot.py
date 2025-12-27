import torch
import xtransformers as xt

def initialize_decoder():
    decoder = xt.TransformerWrapper(
        num_tokens=2049,
        max_seq_len=500,
        use_abs_pos_emb=True,
        scaled_sinu_pos_emb=True,
        attn_layers=xt.Decoder(
            dim=1024,
            depth=24,
            heads=16,
            attn_dim_head=64,
            attn_flash=True,
            ff_no_bias=True,
            cross_attend=True,
        ),
    )
    return decoder

def generate_example_inputs():
    decoder_input = torch.randint(0, 2048, (2, 20))  # Decoder input (batch=2, seq_len=20)
    context_input = torch.rand(2, 4, 1024)  # Context input (batch=2, seq_len=4, dim=1024)
    context_mask = torch.zeros(2, 4, dtype=torch.bool)  # All context tokens are "non-padding"
    return decoder_input, context_input, context_mask

def compute_output(decoder, decoder_input, context_input, context_mask):
    output = decoder(decoder_input, context=context_input, context_mask=context_mask)
    return output

def main():
    decoder = initialize_decoder()
    decoder_input, context_input, context_mask = generate_example_inputs()
    output = compute_output(decoder, decoder_input, context_input, context_mask)

    print("Output shape:", output.shape)
    print("Output values:", output)

if __name__ == "__main__":
    main()