import torch
from transformers import T5WithRotaryEmbeddings

def test_rotary_positional_embeddings():
    # Setup minimal example with a small sequence length to trigger the issue
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = T5WithRotaryEmbeddings()
    cos_cached = model.cos_cached
    sin_cached = model.sin_cached
    
    batch_size = 1
    seq_len = 2  # Reduce sequence length to trigger bug
    x = torch.randn(batch_size, seq_len, device=device)
    
    x_rope = (x * cos_cached[:, :, :, :seq_len]) + (-0.5 * x) * sin_cached[:, :, :, :seq_len]
    
    print("Without broadcasting issues:")
    print(x_rope)

test_rotary_positional_embeddings()