import torch
import torch.nn as nn
from torch.autograd import Variable

class PatchDiscriminator(nn.Module):
    """
    Discriminator module with final convolution adjusted for feature map size
    and adaptive pooling to produce a single prediction per sample.
    """
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        self.model = nn.Sequential(
            # ... existing layers ...
            nn.Conv2d(512, 1, kernel_size=3, padding=1),  # Adjusted for correct reduction
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1, 1),
        )

    def forward(self, x):
        return self.model(x)

def make_bce_label(batch_size: int, real: bool = True) -> Variable:
    """
    Create a BCE target tensor wrapped in a Variable.
    Preserves original behavior of using Variable(torch.ones/zeros(batch_size)).
    """
    return Variable(torch.ones(batch_size)) if real else Variable(torch.zeros(batch_size))

if __name__ == "__main__":
    # Example instantiation and label creation
    discriminator = PatchDiscriminator()
    batch_size = 16

    y_real = make_bce_label(batch_size, real=True)
    y_fake = make_bce_label(batch_size, real=False)