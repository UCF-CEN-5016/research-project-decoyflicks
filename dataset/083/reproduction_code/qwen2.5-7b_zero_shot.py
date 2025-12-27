import torch
import torch.nn as nn
import torchvision

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, context):
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        attn = torch.matmul(q, k.transpose(1, 2))  # Transpose k before matrix multiplication
        return self.to_out(attn)

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = 768
        self.patch_emb = nn.Conv2d(3, self.embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.cross_attn = CrossAttention(self.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.GELU(),
            nn.Linear(self.embed_dim * 4, self.embed_dim)
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        self.fc = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_emb(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token, x], dim=1)
        context = x
        x = self.cross_attn(x, context)
        x = self.mlp(x)
        x = self.norm(x)
        return self.fc(x)

# Test the model
model = ViT()
input = torch.randn(1, 3, 224, 224)
output = model(input)