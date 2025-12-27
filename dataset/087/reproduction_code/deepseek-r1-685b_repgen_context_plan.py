import torch
from x_transformers import TransformerWrapper, Decoder

def setup_model():
    model = TransformerWrapper(
        num_tokens=256,
        max_seq_len=512,
        attn_layers=Decoder(
            dim=512,
            depth=6,
            heads=8,
            attn_flash=True,  # Enable flash attention
            attn_alibi=True   # Enable alibi position bias
        )
    ).cuda()
    return model

def generate_custom_alibi(batch_size, num_heads, seq_len):
    return torch.randn(batch_size, num_heads, seq_len, seq_len).cuda()

def main():
    model = setup_model()
    
    batch_size = 2
    seq_len = 128
    custom_alibi = generate_custom_alibi(batch_size, 8, seq_len)

    x = torch.randint(0, 256, (batch_size, seq_len)).cuda()

    try:
        out = model(x, attn_bias=custom_alibi)
        print("Forward pass successful")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Flash attention failed to handle 4D alibi bias")

if __name__ == "__main__":
    main()