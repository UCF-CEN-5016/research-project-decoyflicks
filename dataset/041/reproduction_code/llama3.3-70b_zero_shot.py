import torch
import torch.nn as nn

class DALLE(nn.Module):
    def __init__(self):
        super(DALLE, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, text, image_codes):
        return self.fc(torch.cat((text, image_codes), dim=1))

def train_dalle_batch(dalle, train_data, _, idx, __):
    text, image_codes, mask = train_data
    loss = dalle(text[idx, ...], image_codes[idx, ...], mask=mask[idx, ...], return_loss=True)
    return loss

dalle = DALLE()
train_data = (torch.randn(10, 10), torch.randn(10, 10), torch.randn(10))
idx = 0

try:
    train_dalle_batch(dalle, train_data, None, idx, None)
except TypeError as e:
    print(e)