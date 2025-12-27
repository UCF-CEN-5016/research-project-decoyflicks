import torch

# Create a model encoder with pos_embedding having 5 columns (0-4 indices)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Assume pos_embedding has 5 columns
        self.pos_embedding = nn.Parameter(torch.randn(100, 5))  # dummy tensor

model = Encoder()
encoder = model.encoder
pos_embedding = encoder.pos_embedding

# Example tokens with shape [batch_size, num_tokens]
tokens = torch.randn(32, 5)

# num_p is set to a value that exceeds the available columns in pos_embedding[1: ... ]
num_p = 6  # pos_embedding has only indices 0-4; trying to slice up to (6) would be index out of range

try:
    tokens = tokens + pos_embedding[:, 1:(num_p + 1)]
except IndexError as e:
    print(f"IndexError: {e}")