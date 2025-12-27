import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader
from vit_pytorch.cross_vit import CrossViT

# Set up transformations - resize MNIST to 256x256 while keeping it single-channel
transform = Compose([
    Resize((256, 256)),
    ToTensor(),
])

# Load MNIST dataset (which has grayscale/single-channel images)
dataset = MNIST('./data', train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=1)

# Initialize CrossViT model
model = CrossViT(
    image_size=256,
    num_classes=10,  # MNIST has 10 classes
    depth=2,  # Using smaller depth for faster execution
    sm_dim=192,
    sm_patch_size=16,
    sm_enc_depth=2,
    sm_enc_heads=8,
    sm_enc_mlp_dim=2048,
    lg_dim=384,
    lg_patch_size=64,
    lg_enc_depth=2,
    lg_enc_heads=8,
    lg_enc_mlp_dim=2048,
    cross_attn_depth=1,
    cross_attn_heads=8
)

# Try to process a single-channel image from MNIST
for images, _ in loader:
    try:
        output = model(images)  # This will trigger the error
        print("Successfully processed image!")
    except RuntimeError as e:
        print(f"Error: {e}")
    break