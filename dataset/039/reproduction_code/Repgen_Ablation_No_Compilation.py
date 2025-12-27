import torch

class RotaryPositionalEmbeddings:
    def __init__(self):
        self.cos_cached = 1.5
        self.sin_cached = -2.0

def test_rotary_positional_embeddings():
    x_rope = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    neg_half_x = -x_rope / 2
    # Ensure that cos_cached and sin_cached have the correct shape to broadcast with x_rope
    cos_cached = torch.tensor(self.cos_cached).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    sin_cached = torch.tensor(self.sin_cached).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    
    # Correctly apply the rotary position embeddings
    x_rope = (x_rope * cos_cached[:x_rope.shape[0]]) + (neg_half_x * sin_cached[:x_rope.shape[0]])

# Run the test
test_rotary_positional_embeddings()