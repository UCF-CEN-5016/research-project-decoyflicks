import torch
import torch.nn as nn

class BertSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(768, 768)
        self.key = nn.Linear(768, 768)
        self.value = nn.Linear(768, 768)
        # Using a different activation function entirely
        self.activation = nn.GELU()
    
    def forward(self, hidden_states):
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        # Using the module implementation instead of functional
        query = self.activation(query)
        return query, key, value

model = BertSelfAttention()
inputs = torch.randn(2, 10, 768)
# This will work without error
q, k, v = model(inputs)
print(f"Query shape: {q.shape}")