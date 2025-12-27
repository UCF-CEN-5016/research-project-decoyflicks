import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

class Attention(nn.Module):
    def __init__(self, num_heads=8):
        super(Attention, self).__init__()
        self.query_linear = nn.Linear(128, 128)
        self.key_linear = nn.Linear(128, 128)
        self.value_linear = nn Linear(128, 128)
        self.num_heads = num_heads

    def forward(self, query, key, value):
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        attention = F.einsum('bijd,jd->bid', query, key)  # (batch_size, seq_len, num_heads, dim)
        attention = rearrange(attention, 'b i d -> b (i d)')
        attention = reduce(attention, 'b i j -> b i', reduction='softmax')

        output = F.einsum('bijd,jd->bid', value, attention)  # (batch_size, seq_len, dim)
        return output

# Sample input
query = torch.randn(1, 10, 128)  # (batch_size, seq_len, dim)
key = torch.randn(1, 10, 128)  # (batch_size, seq_len, dim)
value = torch.randn(1, 10, 128)  # (batch_size, seq_len, dim)

# Call the Attention module
attn = Attention()
output = attn(query, key, value)

# Check the attention tensor
print(attn)
print(output.shape)  # (batch_size, seq_len, dim)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

class Attention(nn.Module):
    def __init__(self, num_heads=8):
        super(Attention, self).__init__()
        self.query_linear = nn.Linear(128, 128)
        self.key_linear = nn.Linear(128, 128)
        self.value_linear = nn.Linear(128, 128)
        self.num_heads = num_heads

    def forward(self, query, key, value):
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        attention = F.einsum('bijd,jd->bid', query, key)  # (batch_size, seq_len, num_heads, dim)
        attention = rearrange(attention, 'b i d -> b (i d)')
        attention = reduce(attention, 'b i j -> b i', reduction='softmax')

        output = F.einsum('bijd,jd->bid', value, attention)  # (batch_size, seq_len, dim)
        return output

# Sample input
query = torch.randn(1, 10, 128)  # (batch_size, seq_len, dim)
key = torch.randn(1, 10, 128)  # (batch_size, seq_len, dim)
value = torch.randn(1, 10, 128)  # (batch_size, seq_len, dim)

# Call the Attention module
attn = Attention()
output = attn(query, key, value)

# Check the attention tensor
print(attn)
print(output.shape)  # (batch_size, seq_len, dim)