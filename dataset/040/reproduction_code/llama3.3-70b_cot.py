import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple Self-Attention module
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # Get the batch size, sequence length, and embed dim
        batch_size, sequence_length, embed_dim = x.size()

        # Reshape the input for multi-head attention
        query = self.query_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute the attention scores
        attention_scores = torch.einsum("bhld,bhmd->bhlm", query, key)

        # Apply softmax along the wrong axis (bug)
        # attention_weights = attention_scores.softmax(dim=1)  # Wrong axis
        attention_weights = attention_scores.softmax(dim=2)  # Corrected axis

        # Apply attention weights to the value
        attention_output = torch.einsum("bhlm,bhmd->bhld", attention_weights, value)

        # Reshape the output
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, embed_dim)

        return attention_output

# Minimal setup
if __name__ == "__main__":
    # Set the seed for reproducibility
    torch.manual_seed(42)

    # Create a random input tensor
    batch_size = 32
    sequence_length = 10
    embed_dim = 128
    num_heads = 8
    x = torch.randn(batch_size, sequence_length, embed_dim)

    # Create a Self-Attention module
    attention_module = SelfAttention(embed_dim, num_heads)

    # Apply the Self-Attention module
    output = attention_module(x)

    print(output.shape)