class CrossViT(nn.Module):
    def __init__(self, img_size=256, patch_size=16, num_classes=1000,
                 embed_dim=768, hidden_dim=384, mlp_dim=1536, qkv_dim=96):
        super().__init__()
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim

        # Vision Encoders (from Vision transformer)
        self.vit_base = VisionTransformer(
            img_size=img_size, patch_size=patch_size,
            num_classes=num_classes, embed_dim=embed_dim,
            hidden_dim=hidden_dim, mlp_dim=mlp_dim, qkv_dim=qkv_dim
        )
        self.vit_large = VisionTransformer(
            img_size=img_size, patch_size=patch_size,
            num_classes=num_classes, embed_dim=embed_dim,
            hidden_dim=hidden_dim, mlp_dim=mlp_dim, qkv_dim=qkv_dim
        )

    def forward(self, x):
        # Get image embeddings for small and large patches respectively
        sm_tokens = self.sm_image_embedder(x)  # shape [batch_size, n_small, embed_dim]
        lg_tokens = self.lg_image_embedder(x)  # shape [batch_size, n_large, embed_dim]

        # Multi-scale attention layer
        x = self.multi_scale_encoder(sm_tokens, lg_tokens)

        # Concatenate tokens from small and large patches along the token dimension
        x = torch.cat(x, dim=1)  # Shape becomes [batch_size, (n_small + n_large), embed_dim]

        # Project to class logits
        x = self.mlp_head(x)  # shape [batch_size, num_classes]
        return x