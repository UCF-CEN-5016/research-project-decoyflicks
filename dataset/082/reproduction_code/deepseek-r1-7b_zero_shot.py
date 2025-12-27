import torch
from vit_pytorch import ViT
from vit_pytorch.mpp import MPP

# Initialize model with correct dimensions
model = ViT(
    image_size=256,
    patch_size=32,
    num_classes=1000,
    dim=1024,       # Token embedding dimension (e.g., 1024)
    depth=6,
    heads=8,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)

# MPP configuration parameters
mask_prob = 0.15
random_patch_prob = 0.30
replace_prob = 0.50

mpp_trainer = MPP(
    transformer=model,
    patch_size=32,
    dim=1024,       # Feature dimension after embedding (e.g., 1024)
    mask_prob=mask_prob,
    random_patch_prob=random_patch_prob,
    replace_prob=replace_prob
)

opt = torch.optim.Adam(mpp_trainer.parameters(), lr=3e-4)

def sample_unlabelled_images():
    return torch.FloatTensor(20, 3, 256, 256).uniform_(0., 1.)

for _ in range(100):
    images = sample_unlabelled_images()
    
    # Apply MPP forward pass
    masked_tokens = mpp_trainer(images)
    
    # No need to zero_grad or backward here as loss is from MPP
    
# Save the model weights without issues
torch.save(model.state_dict(), './pretrained-net.pt')