import torch
from torch import nn

class DummyDALLE(nn.Module):
    def forward(self, text, image_codes):
        return torch.tensor(0.0)

dalle = DummyDALLE()
text = torch.randint(0, 10, (1, 10))
image_codes = torch.randint(0, 10, (1, 10))
mask = torch.ones(1, 10, dtype=torch.bool)

loss = dalle(text, image_codes, mask=mask, return_loss=True)