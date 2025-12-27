import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # 128×128 → 64×64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # 64×64 → 32×32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),  # 32×32 → 16×16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),  # 16×16 → 8×8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),  # Flatten to 1D
            nn.Linear(512 * 8 * 8, 1)  # Final output is 1D, shape (batch_size, 1)
        )

    def forward(self, x):
        return self.main(x)

# Example usage
model = Discriminator()
input_tensor = torch.randn(128, 3, 128, 128)  # Batch size 128, 128×128 images
output = model(input_tensor)
print(output.shape)  # Should be (128, 1)