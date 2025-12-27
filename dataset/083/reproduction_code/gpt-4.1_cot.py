import torch
from torch import nn

# Minimal PatchEmbedding module similar to what's used in CrossViT
class PatchEmbed(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, patch_size=8):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim), N=H*W
        x = self.norm(x)
        return x

# Minimal cross attention between two token sets, similar to CrossViT fusion
class CrossAttention(nn.Module):
    def __init__(self, dim_small, dim_large):
        super().__init__()
        self.to_q = nn.Linear(dim_small, dim_small, bias=False)
        self.to_k = nn.Linear(dim_large, dim_small, bias=False)
        self.to_v = nn.Linear(dim_large, dim_small, bias=False)
        self.to_out = nn.Linear(dim_small, dim_small)

    def forward(self, tokens_small, tokens_large):
        # tokens_small: (B, N_small, dim_small)
        # tokens_large: (B, N_large, dim_large)
        q = self.to_q(tokens_small)  # (B, N_small, dim_small)
        k = self.to_k(tokens_large)  # (B, N_large, dim_small)
        v = self.to_v(tokens_large)  # (B, N_large, dim_small)

        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (q.shape[-1] ** 0.5), dim=-1)
        out = torch.bmm(attn, v)  # (B, N_small, dim_small)
        return self.to_out(out)

# Minimal CrossViT forward pass that causes the bug
class MinimalCrossViT(nn.Module):
    def __init__(self):
        super().__init__()
        # Two patch embeddings with different patch sizes and dims
        self.patch_embed_small = PatchEmbed(in_channels=3, embed_dim=64, patch_size=8)
        self.patch_embed_large = PatchEmbed(in_channels=3, embed_dim=128, patch_size=16)
        
        # Cross attention module (intentionally mismatched dims to trigger bug)
        self.cross_attn = CrossAttention(dim_small=64, dim_large=128)

    def forward(self, x):
        # x: (B, 3, H, W)
        tokens_small = self.patch_embed_small(x)  # (B, N_small, 64)
        tokens_large = self.patch_embed_large(x)  # (B, N_large, 128)

        # Intentionally cause shape mismatch by feeding tokens with incompatible batch sizes or shapes
        out = self.cross_attn(tokens_small, tokens_large)
        return out

if __name__ == '__main__':
    model = MinimalCrossViT()
    # Create dummy input tensor: batch size 2, RGB image 64x64
    dummy_input = torch.randn(2, 3, 64, 64)

    # Run forward pass - should work fine if shapes align
    output = model(dummy_input)
    print(output.shape)

    # Now intentionally trigger a bug by sending input with incorrect size
    # For example, change input size so that patch embeddings produce incompatible sequence lengths
    dummy_input_bug = torch.randn(2, 3, 60, 60)  # Not divisible by patch sizes cleanly
    output_bug = model(dummy_input_bug)  # This may cause shape mismatch during attention