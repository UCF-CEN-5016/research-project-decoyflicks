import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from vector_quantize_pytorch import VectorQuantize

lr = 3e-4
train_iter = 1000
num_codes = 256
seed = 1234
device = "cuda" if torch.cuda.is_available() else "cpu"

class SimpleVQAutoEncoder(nn.Module):
    def __init__(self, implicit_neural_codebook=True, **vq_kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.GELU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # Initialize VectorQuantize only if implicit_neural_codebook is True
                VectorQuantize(dim=32, accept_image_fmap=True, implicit_neural_codebook=implicit_neural_codebook, **vq_kwargs) if implicit_neural_codebook else nn.Identity(),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            ]
        )

    def forward(self, x):
        indices = None  # Initialize indices to avoid potential use before assignment
        commit_loss = 0  # Initialize commit_loss to avoid potential use before assignment
        for layer in self.layers:
            if isinstance(layer, VectorQuantize):
                x, indices, commit_loss = layer(x)
            else:
                x = layer(x)
        return x.clamp(-1, 1), indices, commit_loss

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
train_dataset = DataLoader(
    datasets.FashionMNIST(
        root="~/data/fashion_mnist", train=True, download=True, transform=transform
    ),
    batch_size=256,
    shuffle=True,
)

torch.random.manual_seed(seed)
model = SimpleVQAutoEncoder(implicit_neural_codebook=False, codebook_size=num_codes).to(device)  # Set implicit_neural_codebook to False
opt = torch.optim.AdamW(model.parameters(), lr=lr)

def train(model, train_loader, train_iterations=1000, alpha=10):
    def iterate_dataset(data_loader):
        data_iter = iter(data_loader)
        while True:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                x, y = next(data_iter)
            yield x.to(device), y.to(device)

    for _ in range(train_iterations):
        opt.zero_grad()
        x, _ = next(iterate_dataset(train_loader))
        out, indices, cmt_loss = model(x)
        rec_loss = (out - x).abs().mean()
        (rec_loss + alpha * cmt_loss).backward()
        opt.step()

train(model, train_dataset, train_iterations=train_iter)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.data}")