import torch
import einops

# Custom rotary embedding
class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim):
        super(RotaryEmbedding, self).__init__()
        self.rotary = torch.nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x):
        return rearrange(x + self.rotary, 'n c -> n c')

# Define the model
model = RotaryEmbedding(dim=32)

# Set up input and rotary xpos
input_tensor = torch.randn(1, 32)
rotary_xpos = torch.randn(1, 32)

# Call forward with rotary xpos
output = model(input_tensor + rotary_xpos)

print("Output shape:", output.shape)