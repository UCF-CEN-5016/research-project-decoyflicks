import torch
from vit_pytorch import ViT
from mpp_module import MPP

# Define the ViT model with specified parameters
v = ViT(
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

# Define the MPP model with specific parameters
mpp = MPP(transformer=v, patch_size=32, dim=1024, mask_prob=0.15, random_patch_prob=0.30, replace_prob=0.50)

# Initialize the Adam optimizer with a learning rate of 3e-4
opt = torch.optim.Adam(mpp.parameters(), lr=3e-4)

def sample_unlabelled_images(batch_size=20, height=256, width=256, channels=3):
    return torch.randn(batch_size, channels, height, width)

# Training loop for 100 epochs
for epoch in range(100):
    img = sample_unlabelled_images()
    loss = mpp(img)
    
    opt.zero_grad()
    loss.backward()
    opt.step()

# Save the state dictionary of the ViT model to a file named './pretrained-net.pt'
torch.save(v.state_dict(), './pretrained-net.pt')