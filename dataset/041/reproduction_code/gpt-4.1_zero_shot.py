import torch
import torch.nn as nn

class DummyDalle(nn.Module):
    def forward(self, x, y, return_loss=False):
        return (x + y).float().sum() if return_loss else (x + y)

dalle = DummyDalle()
text = torch.randint(0, 10, (4, 5))
image_codes = torch.randint(0, 10, (4, 5))
mask = torch.ones_like(text)

dalle(text, image_codes, mask=mask, return_loss=True)