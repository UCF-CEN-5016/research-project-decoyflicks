import torch
import torch.nn as nn

# Simulate the RotaryPositionalEmbeddings class (simplified)
class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # Simulate an operation that requires even dimensions
        if self.dim % 2 != 0:
            raise ValueError(f"Dimension {self.dim} must be even")
        return x

# Minimal setup to reproduce the bug
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 4)  # 4 features as per the example
        self.rotary_pe = RotaryPositionalEmbeddings(3)  # Incorrect value causing the bug

    def forward(self, x):
        x = self.embed(x)
        return self.rotary_pe(x)

# Trigger the bug
model = SimpleModel()
input_tensor = torch.randint(0, 10, (2, 5))  # Batch size 2, sequence length 5
try:
    output = model(input_tensor)
except ValueError as e:
    print("Error caught:", e)