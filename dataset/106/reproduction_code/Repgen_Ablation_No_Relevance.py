import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import trange
from vector_quantize_pytorch import VectorQuantize

batch_size = 256
image_dimensions = (28, 28)

# Create random uniform input data with shape (batch_size, height, width, channels=1)
x = torch.randn(batch_size, *image_dimensions, 1)

# Normalize the input data using torchvision.transforms.Normalize((0.5,), (0.5,))
transform = Compose([Normalize((0.5,), (0.5,)), ToTensor()])
dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size)

lr = 3e-4
train_iter = 1000
num_codes = 256
seed = 1234
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SimpleVQAutoEncoder(nn.Module):
    def __init__(self, enc_channels, dec_channels, num_codes):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(1, enc_channels[0], 3),
            nn.MaxPool2d((2, 2)),
            nn.GELU(),
            VectorQuantize(num_codes=num_codes, dim=enc_channels[0]),
            nn.ConvTranspose2d(enc_channels[0], dec_channels[0], (2, 2)),
            nn.GELU(),
            nn.ConvTranspose2d(dec_channels[0], 1, 3),
        ])

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, VectorQuantize):
                x, indices, commit_loss = layer(x)
            else:
                x = layer(x)

        return x.clamp(-1, 1), indices, commit_loss

model = SimpleVQAutoEncoder([32, 64], [64, 32], num_codes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for _ in trange(train_iter):
    optimizer.zero_grad()
    x_hat, _, _ = model(x.to(device))
    loss = ((x - x_hat) ** 2).mean().clamp(0, float('inf'))  # Ensure non-negative loss
    if torch.isnan(loss):
        raise AssertionError("NaN values found in the distance calculation")
    loss.backward()
    optimizer.step()