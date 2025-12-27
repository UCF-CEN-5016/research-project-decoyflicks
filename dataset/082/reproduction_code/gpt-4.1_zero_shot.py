import torch
from torch import nn

class DummyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_patch_embedding = nn.Sequential(
            nn.Flatten(2),
            nn.Linear(3*32*32, 1024),
            nn.LayerNorm(1024)
        )

    def forward(self, x):
        return self.to_patch_embedding(x)

class MPP(nn.Module):
    def __init__(self, transformer, patch_size):
        super().__init__()
        self.transformer = transformer
        self.patch_size = patch_size

    def forward(self, img):
        p = self.patch_size
        b, c, h, w = img.shape
        patches = img.unfold(2, p, p).unfold(3, p, p)  # b,c,nh,nw,p,p
        nh, nw = patches.shape[2], patches.shape[3]
        patches = patches.contiguous().view(b, c, nh*nw, p, p)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()  # b,nh*nw,c,p,p
        patches = patches.view(b*nh*nw, c, p, p)
        x = self.transformer.to_patch_embedding(patches)  # expecting (b*nh*nw, 1024)
        x = x.view(b, nh*nw, -1)
        x = self.transformer.to_patch_embedding[-1](x)  # wrong input shape here
        return x.mean()

transformer = DummyTransformer()
mpp = MPP(transformer, patch_size=32)

img = torch.randn(20, 3, 256, 256)
mpp(img)