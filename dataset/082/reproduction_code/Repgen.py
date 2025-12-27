import torch
from vit_pytorch import ViT
from vit_pytorch.mpp import MPP

# Step 1: Initialize the ViT model with specified parameters.
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

# Step 2: Create an MPP trainer using the ViT model.
mpp_trainer = MPP(
    transformer=model,
    patch_size=32,
    dim=1024,
    mask_prob=0.15,          # probability of using token in masked prediction task
    random_patch_prob=0.30,  # probability of randomly replacing a token being used for mpp
    replace_prob=0.50        # probability of replacing a token being used for mpp with the mask token
)

# Step 3: Set up the optimizer.
opt = torch.optim.Adam(mpp_trainer.parameters(), lr=3e-4)

# Step 4: Sample 20 unlabelled images and pass them through the MPP trainer.
def sample_unlabelled_images():
    return torch.FloatTensor(20, 3, 256, 256).uniform_(0., 1.)

for _ in range(100):
    # Step 5: Sample unlabelled images and compute loss.
    images = sample_unlabelled_images()
    loss = mpp_trainer(images)

    # Step 6: Zero out gradients before backpropagation.
    opt.zero_grad()

    # Step 7: Perform the backward pass followed by an optimization step.
    loss.backward()
    opt.step()

# Step 8: Save the state dictionary of the model.
torch.save(model.state_dict(), './pretrained-net.pt')