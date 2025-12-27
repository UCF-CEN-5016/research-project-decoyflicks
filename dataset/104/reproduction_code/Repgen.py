import torch
from torch import nn, einsum
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import trange
import os
import sys

# Add repository to path - you need to clone the repository first
# git clone https://github.com/lucidrains/vector-quantize-pytorch.git
# Then adjust this path to where you cloned it
repo_path = "./vector-quantize-pytorch"
if os.path.exists(repo_path):
    sys.path.append(repo_path)
else:
    raise FileNotFoundError("Repository not found. Please clone it first.")

# Import the ResidualVQ module directly
from vector_quantize_pytorch.residual_vq import ResidualVQ

# Define hyperparameters
batch_size = 64
learning_rate = 1e-3
iterations = 10
dim = 28 * 28  # Flattened image dimension for MNIST

# Load and prepare dataset
transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a simple autoencoder model that uses ResidualVQ
class SimpleVQAutoEncoder(nn.Module):
    def __init__(self, implicit_neural_codebook=True):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # ResidualVQ layer - this is what we're testing
        self.vq = ResidualVQ(
            dim = 256,
            num_quantizers = 2,
            codebook_size = 256,
            commitment_weight = 0.25,
            decay = 0.99,
            kmeans_init = True,
            kmeans_iters = 10,
            threshold_ema_dead_code = 2,
            implicit_neural_codebook = implicit_neural_codebook  # This is the parameter in question
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh(),
            nn.Unflatten(1, (1, 28, 28))
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        quantized, indices, commit_loss = self.vq(encoded)
        reconstructed = self.decoder(quantized)
        
        # Calculate reconstruction loss
        recon_loss = torch.mean((reconstructed - x) ** 2)
        total_loss = recon_loss + commit_loss
        
        return reconstructed, indices, total_loss

# Initialize model with implicit_neural_codebook=False
model = SimpleVQAutoEncoder(implicit_neural_codebook=False).to(device)

# Before training - check if MLPs exist and are initialized
print("MLP parameter check before training:")
mlp_params = [name for name, _ in model.named_parameters() if 'mlp' in name]
for name in mlp_params:
    print(f"Found MLP parameter: {name}")

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for _ in trange(iterations):
    for data, _ in dataloader:
        inputs = data.to(device)
        
        # Forward pass
        reconstructions, _, loss = model(inputs)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Validate MLPs initialization
print("\nChecking MLP parameters after training:")
mlp_params = [(name, param) for name, param in model.named_parameters() if 'mlp' in name]

if len(mlp_params) > 0:
    print(f"Found {len(mlp_params)} MLP parameters despite implicit_neural_codebook=False:")
    for name, param in mlp_params:
        non_zero = (param != 0).sum().item()
        total = param.numel()
        print(f"- {name}: {non_zero}/{total} non-zero values, requires_grad={param.requires_grad}")
    print("\nBUG CONFIRMED: MLPs are initialized despite implicit_neural_codebook=False")
else:
    print("No MLP parameters found. The bug doesn't appear to be present.")

# Print the structure of the ResidualVQ module for reference
print("\nResidualVQ structure:")
for name, module in model.vq.named_modules():
    if name:
        print(f"- {name}")