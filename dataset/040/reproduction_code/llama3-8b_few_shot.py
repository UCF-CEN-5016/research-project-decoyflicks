import torch
from einops import rearrange, reduce

# Minimal UNet architecture with attention module (buggy)
class AttentionBlock(torch.nn.Module):
    def __init__(self):
        super(AttentionBlock, self).__init__()
        self.query_linear = torch.nn.Linear(128, 128)
        self.key_linear = torch.nn.Linear(128, 128)
        self.value_linear = torch.nn.Linear(128, 128)

    def forward(self, x):
        q = rearrange(self.query_linear(x), 'b h w c -> b (h w) c')
        k = rearrange(self.key_linear(x), 'b h w c -> b (h w) c')
        v = rearrange(self.value_linear(x), 'b h w c -> b (h w) c')

        attention = torch.matmul(q, k.transpose(-1, -2))
        attention = reduce(attention, 'b n d -> b n', reduction='softmax')  # <--- Incorrect calculation

        out = attention * v
        return rearrange(out, 'b (h w) c -> b h w c')

# Example usage
model = AttentionBlock()
input_tensor = torch.randn(1, 128, 28, 28)
output = model(input_tensor)

print("Output shape:", output.shape)