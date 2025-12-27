import torch
import torch.nn as nn
from typing import Any

class SimpleAttention(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super(SimpleAttention, self).__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query_input: torch.Tensor, key_input: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(query_input).unsqueeze(-1)
        key = self.key_proj(key_input).unsqueeze(1)
        attn = torch.einsum('bij,bkj->bkj', query, key)
        return attn

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.attn = SimpleAttention(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        return self.attn(x, y)

if __name__ == "__main__":
    net = Net()
    input_tensor = torch.randn(1, 4, 16, 16)
    output = net(input_tensor)
    print(output.shape)