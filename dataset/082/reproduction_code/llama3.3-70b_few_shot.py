import torch
from vit_pytorch import ViT
from vit_pytorch.mpp import MPP

# Model setup
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
)

# MPP setup with incorrect dim
mpp_trainer = MPP(
    transformer=model,
    patch_size=32,
    dim=1024,
    mask_prob=0.15,          # probability of using token in masked prediction task
    random_patch_prob=0.30,  # probability of randomly replacing a token being used for mpp
    replace_prob=0.50,       # probability of replacing a token being used for mpp with the mask token
)

# Sample data
def sample_unlabelled_images():
    return torch.FloatTensor(20, 3, 256, 256).uniform_(0., 1.)

# Training loop that produces dimension mismatch error
for _ in range(1):
    images = sample_unlabelled_images()
    loss = mpp_trainer(images)
    print(f"Loss: {loss}")