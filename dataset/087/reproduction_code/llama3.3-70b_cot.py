import torch
from x_transformers import TransformerLayer, Attention

# Set up minimal environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a simple transformer layer
class SimpleTransformerLayer(TransformerLayer):
    def __init__(self, dim, num_heads):
        super().__init__(dim=dim, num_heads=num_heads, dim_head=dim//num_heads, attn_dropout=0.0, ff_dropout=0.0)

# Create a simple transformer layer
layer = SimpleTransformerLayer(dim=256, num_heads=8).to(device)

# Create a batch of input tensors
batch_size = 32
seq_len = 100
input_tensor = torch.randn(batch_size, seq_len, 256).to(device)

# Set up custom alibi positions
custom_positions = torch.randn(batch_size, seq_len, 256).to(device)

# Trigger the bug by setting attn_flash=True
attention = Attention(dim=256, num_heads=8, attn_flash=True)

# Try to apply the attention mechanism with custom positions
try:
    output = attention(input_tensor, custom_positions)
    print("Attention applied successfully")
except Exception as e:
    print("Error applying attention:", str(e))