import torch
from vit_pytorch import ViT
from vit_pytorch.mpp import MPP

# Model setup
model = ViT(
    image_size=256,
    patch_size=32,
    num_classes=1000,
    dim=1024,          # This is the hidden dimension
    depth=6,
    heads=8,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)

# MPP Trainer with correct dimensions
mpp_trainer = MPP(
    transformer=model,
    patch_size=32,
    dim=1024,          # Must match model's dim
    channels=3,        # Important for correct patch dimension calculation
    mask_prob=0.15,
    random_patch_prob=0.30,
    replace_prob=0.50,
)

opt = torch.optim.Adam(mpp_trainer.parameters(), lr=3e-4)

def sample_unlabelled_images():
    return torch.randn(20, 3, 256, 256)  # Changed to randn for more realistic values

for _ in range(100):
    images = sample_unlabelled_images()
    loss = mpp_trainer(images)
    opt.zero_grad()
    loss.backward()
    opt.step()

torch.save(model.state_dict(), './pretrained-net.pt')