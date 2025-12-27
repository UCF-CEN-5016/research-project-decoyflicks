def generate_attends(alibi: torch.Tensor):
    if isinstance(alibi, AlibiPositionCausal):
        pos = alibi.pos
    else:
        pos = alibi

    # Add padding to positions for flash attention
    batch_size, seq_len, _, num_heads = pos.size()
    pos = pos.view(batch_size * num_heads, -1, seq_len)
    pos += torch.arange(seq_len).to(pos.device) * seq_len

import torch
from x_transformers attends import AlibiPositionCausal, generate_attends

# Unstable training setup
model = torch.nn.Sequential(
    torch.nn.Linear(10, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 10)
)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Sample data
X = torch.randn(32, 10)
y = torch.randn(32, 10)

# Custom alibi with flash attention setup
class CustomAlibi(torch.Tensor):
    def __init__(self, tensor):
        super().__init__(tensor)
        
alibi = AlibiPositionCausal(
    num_heads=8,
    seq_len=50,
    device='cuda'
)
alibi_pos = torch.randn(32, 16, 16, 8)  # Custom alibi shape: [batch, seq, seq, heads]
alibi_pos = CustomAlibi(alibi_pos)

# This should now work with flash attention
result = model(X) @ y.T

print(f"Result shape: {result.shape}")