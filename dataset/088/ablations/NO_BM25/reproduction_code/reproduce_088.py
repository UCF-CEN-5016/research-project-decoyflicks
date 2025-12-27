import torch
import torch.nn as nn
import torch.nn.functional as F

class TestModel(nn.Module):
    def __init__(self, num_tokens, dim):
        super().__init__()
        self.embed_x = nn.Embedding(num_tokens, dim)
        self.embed_y = nn.Embedding(num_tokens, dim)

    def forward(self, x, y, pad_id):
        x_embedded = self.embed_x(x)
        y_embedded = self.embed_y(y)
        aligned = self.align_right(x_embedded, y_embedded, pad_id)
        return aligned

    def align_right(self, x, y, pad_id):
        # Bug: Using hardcoded value=0 instead of pad_id
        return F.pad(x, (0, 0, 0, 1), value=0)

# Reproduction steps
batch_size = 4
num_tokens = 10
dim = 16
x = torch.randint(0, num_tokens, (batch_size, 10))
y = torch.randint(0, num_tokens, (batch_size, 10))

model = TestModel(num_tokens, dim)
pad_id = 0
output = model.forward(x, y, pad_id)

print(output)