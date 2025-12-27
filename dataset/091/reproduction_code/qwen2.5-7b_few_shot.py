import torch
from x_transformers import Transformer

def create_transformer_model():
    model = Transformer(
        dim=64,
        depth=2,
        heads=4,
        ff_mult=4,
        max_seq_len=256,
        num_tokens=1000
    ).to("cuda")
    return model

def generate_output(model, src):
    try:
        with torch.no_grad():
            output = model.generate(
                src,
                start_tokens=torch.tensor([1]),  # Start token
                seq_len=200,  # Decoding sequence length = 200 < 256
                temperature=0.7
            )
            return output
    except RuntimeError as e:
        print("Error occurred:", str(e))
        return None

if __name__ == "__main__":
    model = create_transformer_model()

    # Sample input sequence (longer than target output)
    src = torch.randint(0, 1000, (1, 256))  # Encoding sequence length = 256

    output = generate_output(model, src)
    if output is not None:
        print("Generated output:", output.shape)