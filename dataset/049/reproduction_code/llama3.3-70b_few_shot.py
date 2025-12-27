import torch
import torch.nn as nn
import torch.nn.functional as F

class RelPositionMultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, embed_dim):
        super(RelPositionMultiHeadedAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.qkv_linear = nn.Linear(embed_dim, embed_dim * 3)
        # Uninitialized bias terms
        self.u = torch.Tensor(num_heads, embed_dim)
        self.v = torch.Tensor(num_heads, embed_dim)

    def forward(self, query, key, value):
        qkv = self.qkv_linear(query)
        q, k, v = torch.split(qkv, self.embed_dim, dim=-1)
        q = q.view(-1, self.num_heads, self.embed_dim).transpose(0, 1)
        k = k.view(-1, self.num_heads, self.embed_dim).transpose(0, 1)
        v = v.view(-1, self.num_heads, self.embed_dim).transpose(0, 1)
        # Using uninitialized bias terms
        q = q + self.u[:, None, :]
        k = k + self.v[:, None, :]
        attention_weights = F.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.embed_dim), dim=-1)
        output = torch.matmul(attention_weights, v)
        return output

# Sample usage
if __name__ == "__main__":
    import math
    embed_dim = 128
    num_heads = 8
    batch_size = 32
    sequence_length = 100

    attention = RelPositionMultiHeadedAttention(num_heads, embed_dim)
    query = torch.randn(batch_size, sequence_length, embed_dim)
    key = torch.randn(batch_size, sequence_length, embed_dim)
    value = torch.randn(batch_size, sequence_length, embed_dim)

    output = attention(query, key, value)
    print("Output shape:", output.shape)
    print("Output sum:", output.sum())