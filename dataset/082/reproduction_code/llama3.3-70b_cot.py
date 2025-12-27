import torch
from vit_pytorch import ViT
from vit_pytorch.mpp import MPP

# Set up minimal environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a ViT model
model = ViT(
    image_size=256,
    patch_size=32,
    num_classes=1000,
    dim=1024,
    depth=6,
    heads=8,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
).to(device)

# Create an MPP trainer
mpp_trainer = MPP(
    transformer=model,
    patch_size=32,
    dim=1024,
    mask_prob=0.15,
    random_patch_prob=0.30,
    replace_prob=0.50
).to(device)

# Add triggering conditions
def sample_unlabelled_images():
    return torch.FloatTensor(20, 3, 256, 256).uniform_(0., 1.).to(device)

# Trigger the error
images = sample_unlabelled_images()
loss = mpp_trainer(images)
print(loss)