import torch
import torch.nn as nn

class AttentionBug(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        Q = x
        K = x
        attn = torch.einsum('b h s d, b h s d -> b h s s', Q, K)
        attn = attn.softmax(dim=2)
        return attn

x = torch.randn(1, 2, 3, 3)
model = AttentionBug()
output = model(x)
print(output)