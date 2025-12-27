import torch
import torch.nn as nn

class BertLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768, 768)
    
    def gelu(self, x):
        # Same error, embedded in a BERT-like layer
        return torch.nn.functional.gelu(x, approximate=True)
    
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states

model = BertLayer()
input_tensor = torch.randn(2, 10, 768)
output = model(input_tensor)  # Will raise TypeError