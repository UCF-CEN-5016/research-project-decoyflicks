import torch
from vit_pytorch.vit import ViT
from vit_pytorch.mpp import MPP

image_size = 256
patch_size = 32
num_classes = 1000
dim = 1024
depth = 6
heads = 8
mlp_dim = 2048
dropout = 0.1
emb_dropout = 0.1

model = ViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout, emb_dropout=emb_dropout)
mpp_trainer = MPP(model, patch_size=patch_size, dim=dim, mask_prob=0.15, random_patch_prob=0.30, replace_prob=0.50)

batch_size = 20

def sample_unlabelled_images():
    return torch.rand(batch_size, 3, 256, 256)

opt = torch.optim.Adam(model.parameters(), lr=1e-4)

for _ in range(100):
    images = sample_unlabelled_images()
    loss = mpp_trainer(images)
    opt.zero_grad()
    loss.backward()
    opt.step()

torch.save(model.state_dict(), './pretrained-net.pt')