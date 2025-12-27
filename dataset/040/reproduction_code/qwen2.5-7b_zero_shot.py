import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
    
    def forward(self, x):
        Q = x
        K = x
        attn = torch.matmul(Q, K.transpose(-2, -1))
        attn = nn.functional.softmax(attn, dim=-1)
        return attn

x = torch.randn(1, 2, 3, 3)
model = Attention()
output = model(x)
print(output)