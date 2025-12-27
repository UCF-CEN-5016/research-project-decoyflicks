import torch
import torch.nn as nn

class CrossViT(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        num_classes: int = 1000,
        embed_dim: int = 768,
        hidden_dim: int = 384,
        mlp_dim: int = 1536,
        qkv_dim: int = 96,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim

        # Vision encoders (from Vision transformer)
        self.vit_base = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            num_classes=num_classes,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            qkv_dim=qkv_dim,
        )
        self.vit_large = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            num_classes=num_classes,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            qkv_dim=qkv_dim,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Obtain image embeddings for small and large patches
        small_tokens = self.sm_image_embedder(images)  # [batch, n_small, embed_dim]
        large_tokens = self.lg_image_embedder(images)  # [batch, n_large, embed_dim]

        # Multi-scale attention encoder
        encoded_tokens = self.multi_scale_encoder(small_tokens, large_tokens)

        # Concatenate tokens from both scales along the token dimension
        combined_tokens = torch.cat(encoded_tokens, dim=1)  # [batch, n_small + n_large, embed_dim]

        # Project to class logits
        logits = self.mlp_head(combined_tokens)  # [batch, num_classes]
        return logits