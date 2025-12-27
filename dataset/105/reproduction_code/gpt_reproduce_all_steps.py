import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import trange
from math import log2
from vector_quantize_pytorch import LFQ

seed = 1234
torch.random.manual_seed(seed)

batch_size = 256
codebook_size = 2 ** 8
Z_CHANNELS = 512
NUM_QUANTIZERS = 2
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.FashionMNIST(root="~/data/fashion_mnist", train=True, download=True, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class LFQAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        quantize_dim = int(log2(codebook_size))

        self.encode = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.GroupNorm(4, 32, affine=False),
            nn.Conv2d(32, quantize_dim, kernel_size=1),
        )

        self.quantize = LFQ(
            dim=quantize_dim,
            num_quantizers=NUM_QUANTIZERS,
            codebook_size=codebook_size,
            stochastic_sample_codes=True,
            shared_codebook=True,
            commitment_weight=1.0,
            kmeans_init=True,
            threshold_ema_dead_code=2,
            quantize_dropout=True,
            quantize_dropout_cutoff_index=1,
            quantize_dropout_multiple_of=1,
        )

        self.decode = nn.Sequential(
            nn.Conv2d(quantize_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.encode(x)
        x, indices, entropy_aux_loss = self.quantize(x)
        x = self.decode(x)
        return x.clamp(-1, 1), indices, entropy_aux_loss

model = LFQAutoEncoder().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

def iterate_loader(loader):
    while True:
        for batch in loader:
            yield batch

data_iter = iterate_loader(train_loader)

for _ in trange(1000):
    opt.zero_grad()
    x, _ = next(data_iter)
    x = x.to(device)

    try:
        out, indices, entropy_aux_loss = model(x)
    except RuntimeError as e:
        if "shape mismatch" in str(e):
            embed = model.quantize._codebook.embed
            batch_samples = x
            # attempt to get batch_samples and mask from internal state is complex,
            # so just print shapes involved
            print("RuntimeError:", e)
            print("embed shape:", embed.shape)
            break
        else:
            raise

    rec_loss = F.l1_loss(out, x)
    (rec_loss + entropy_aux_loss).backward()
    opt.step()