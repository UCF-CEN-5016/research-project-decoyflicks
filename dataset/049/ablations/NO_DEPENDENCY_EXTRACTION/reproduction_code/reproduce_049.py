import torch
from fairseq.models import RelPositionMultiHeadedAttention

torch.manual_seed(42)
num_heads = 8
embed_dim = 64
seq_length = 10

input_tensor = torch.randn(1, seq_length, embed_dim)
relative_position_encodings = torch.randn(seq_length, seq_length)

class CustomRelPositionMultiHeadedAttention(RelPositionMultiHeadedAttention):
    def __init__(self, num_heads, embed_dim, relative_position_encodings):
        super().__init__(num_heads, embed_dim, relative_position_encodings)
        self.u = torch.Tensor(1, 1)  # Uninitialized bias parameter
        self.v = torch.Tensor(1, 1)  # Uninitialized bias parameter

attention_layer = CustomRelPositionMultiHeadedAttention(num_heads, embed_dim, relative_position_encodings)
output = attention_layer(input_tensor)

if torch.isnan(output).any():
    print("Output contains NaN values.")