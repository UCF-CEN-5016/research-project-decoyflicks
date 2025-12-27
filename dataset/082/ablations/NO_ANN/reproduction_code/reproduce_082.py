import torch
from vit_pytorch import ViT
from vit_pytorch.mpp import MPP

# Model parameters
image_size = 256
patch_size = 32
num_classes = 1000
dim = 1024
depth = 6
heads = 8
mlp_dim = 2048
dropout = 0.1
emb_dropout = 0.1

# Initialize the ViT model
model = ViT(
    image_size=image_size,
    patch_size=patch_size,
    num_classes=num_classes,
    dim=dim,
    depth=depth,
    heads=heads,
    mlp_dim=mlp_dim,
    dropout=dropout,
    emb_dropout=emb_dropout
)

# Initialize the MPP trainer
mpp_trainer = MPP(
    model,
    patch_size=patch_size,
    dim=dim,
    mask_prob=0.15,
    random_patch_prob=0.30,
    replace_prob=0.50
)

batch_size = 20

# Function to sample unlabelled images
def sample_unlabelled_images():
    return torch.rand(batch_size, 3, image_size, image_size)

# Assuming an optimizer 'opt' is defined
# Define a simple optimizer for demonstration purposes
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for _ in range(100):
    images = sample_unlabelled_images()
    loss = mpp_trainer(images)  # This is where the bug can be reproduced
    loss.backward()
    opt.zero_grad()  # Clear gradients
    opt.step()       # Update model parameters

# Save the model state
torch.save(model.state_dict(), './pretrained-net.pt')