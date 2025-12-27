import torch
import torch.nn as nn

# Define a custom exception for invalid dimensions
class InvalidDimensionError(Exception):
    pass

# Refactored RotaryPositionalEmbeddings class
class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim % 2 != 0:
            raise InvalidDimensionError(f"Dimension {dim} must be even")
        self.dim = dim

    def forward(self, x):
        return x

# Refactored SimpleModel class
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 4)  # 4 features as per the example
        try:
            self.rotary_pe = RotaryPositionalEmbeddings(3)  # Incorrect value causing the bug
        except InvalidDimensionError as e:
            print("Error caught:", e)

    def forward(self, x):
        x = self.embed(x)
        return self.rotary_pe(x) if hasattr(self, 'rotary_pe') else x

# Instantiate the model and handle the error
model = SimpleModel()
input_tensor = torch.randint(0, 10, (2, 5))  # Batch size 2, sequence length 5
output = model(input_tensor)
print("Output shape:", output.shape)