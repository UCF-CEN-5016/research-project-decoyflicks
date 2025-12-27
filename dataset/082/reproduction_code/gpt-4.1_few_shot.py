import torch
from vit_pytorch import ViT
from vit_pytorch.mpp import MPP

# Setup ViT with image size 256 and patch size 32
model = ViT(
    image_size=256,
    patch_size=32,
    num_classes=1000,
    dim=1024,      # Embedding dimension expected by transformer
    depth=6,
    heads=8,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)

# MPP trainer wraps the ViT model
mpp_trainer = MPP(
    transformer=model,
    patch_size=32,
    dim=1024,
    mask_prob=0.15,
    random_patch_prob=0.30,
    replace_prob=0.50,
)

opt = torch.optim.Adam(mpp_trainer.parameters(), lr=3e-4)

def sample_unlabelled_images():
    return torch.FloatTensor(20, 3, 256, 256).uniform_(0., 1.)

# This loop triggers the dimension mismatch error
for _ in range(1):
    images = sample_unlabelled_images()
    loss = mpp_trainer(images)
    opt.zero_grad()
    loss.backward()
    opt.step()