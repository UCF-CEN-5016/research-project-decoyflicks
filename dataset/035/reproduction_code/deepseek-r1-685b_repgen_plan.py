from labml_nn.transformers.rope import RotaryPositionalEmbeddings
import torch

def test_rotary_positional_embeddings(feature_dim):
    try:
        rotary_pe = RotaryPositionalEmbeddings(feature_dim)
        x = torch.randn(1, feature_dim)
        output = rotary_pe(x)
        print(f"Successful output shape: {output.shape}")
    except Exception as e:
        print(f"Error with dim={feature_dim}: {e}")

# Test with correct feature dimension
test_rotary_positional_embeddings(4)

# Test with incorrect feature dimension
test_rotary_positional_embeddings(3)