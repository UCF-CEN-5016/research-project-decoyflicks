import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)

class NaViT_Nest_Tensor_3D(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.norm(x)
        return x

model = NaViT_Nest_Tensor_3D(patch_size=(1, 16, 16), embed_dim=1024)
input = torch.randn(5, 1, 16, 16, 16)
output = model(input)
loss = output.sum()
loss.backward()