import torch
import torch.nn as nn

class MultiHeadedAttention(nn.Module):
    def __init__(self, u_dim=10, v_dim=10):
        super().__init__()
        self.u_bias = torch.Tensor(u_dim)
        self.v_bias = torch.Tensor(v_dim)

    def forward(self, x):
        output = x + self.u_bias + self.v_bias
        return output

u_dim = 10
v_dim = 10
u_bias = torch.Tensor(u_dim)
v_bias = torch.Tensor(v_dim)
print("u_bias:", u_bias)
print("v_bias:", v_bias)

input_tensor = torch.rand(1, 10, 10)
attention_layer = MultiHeadedAttention()
output = attention_layer(input_tensor)

assert not torch.isnan(output).any(), "NaN values detected in output"