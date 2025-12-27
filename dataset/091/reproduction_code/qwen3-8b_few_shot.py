import torch
from x_transformers import Transformer

# Create a transformer model with encoder-decoder architecture
model = Transformer(
    dim=64,
    depth=2,
    heads=4,
    ff_mult=4,
    max_seq_len=256,
    num_tokens=1000
).to("cuda")

# Sample input sequence (longer than target output)
src = torch.randint(0, 1000, (1, 256))  # Encoding sequence length = 256

# Attempt to generate output with shorter sequence length
with torch.no_grad():
    try:
        output = model.generate(
            src,
            start_tokens=torch.tensor([1]),  # Start token
            seq_len=200,  # Decoding sequence length = 200 < 256
            temperature=0.7
        )
        print("Generated output:", output.shape)
    except RuntimeError as e:
        print("Error occurred:", str(e))