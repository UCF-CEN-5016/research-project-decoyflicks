import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignModel(nn.Module):
    def __init__(self):
        super(AlignModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        x = F.softmax(x, dim=-1)
        x = self.fc(x)
        return x

class ForcedAlign(nn.Module):
    def __init__(self):
        super(ForcedAlign, self).__init__()
        self.align_model = AlignModel()

    def forward(self, audio, text):
        log_probs = torch.randn(1, 10, 2)  # incorrect shape
        return self.align_model(log_probs)

def main():
    audio = torch.randn(1, 10)
    text = torch.randn(1, 10)
    forced_align = ForcedAlign()
    output = forced_align(audio, text)

if __name__ == "__main__":
    main()