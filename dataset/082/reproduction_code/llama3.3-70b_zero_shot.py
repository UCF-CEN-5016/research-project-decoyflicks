import torch
from torch import nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embedding = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.patch_embedding(x).flatten(2).transpose(1, 2)

class MPP(nn.Module):
    def __init__(self, transformer, patch_size, dim, mask_prob, random_patch_prob, replace_prob):
        super().__init__()
        self.transformer = transformer
        self.patch_size = patch_size
        self.dim = dim
        self.mask_prob = mask_prob
        self.random_patch_prob = random_patch_prob
        self.replace_prob = replace_prob
        self.to_patch_embedding = nn.Sequential(
            PatchEmbed(patch_size, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        masked_input = x
        masked_input = self.to_patch_embedding(masked_input)
        output = self.transformer(masked_input)
        return output.mean()

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, dropout, emb_dropout):
        super().__init__()
        self.to_patch_embedding = nn.Sequential(
            PatchEmbed(patch_size, dim),
            nn.LayerNorm(dim)
        )
        self.transformer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        x = self.transformer(x)
        return x

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

mpp_trainer = MPP(
    transformer=model.transformer,
    patch_size=32,
    dim=1024,
    mask_prob=0.15,
    random_patch_prob=0.30,
    replace_prob=0.50
)

opt = torch.optim.Adam(mpp_trainer.parameters(), lr=3e-4)

def sample_unlabelled_images():
    return torch.FloatTensor(20, 3, 256, 256).uniform_(0., 1.)

for _ in range(100):
    images = sample_unlabelled_images()
    loss = mpp_trainer(images)
    opt.zero_grad()
    loss.backward()
    opt.step()

torch.save(model.state_dict(), './pretrained-net.pt')