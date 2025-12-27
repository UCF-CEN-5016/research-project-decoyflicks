import torch
import torch.nn as nn


def compute_num_patches(image_size: int, patch_size: int) -> int:
    """
    Compute the number of patches given the image size and patch size.
    """
    return (image_size // patch_size) ** 2


def compute_patch_dim(patch_size: int) -> int:
    """
    Compute the flattened patch dimension (height * width).
    """
    return patch_size ** 2


class VisionTransformer(nn.Module):
    """
    Minimal Vision Transformer-like module that accepts inputs shaped
    (batch_size, num_patches, patch_dim) and projects each patch to an embedding.
    This preserves the original attribute naming and core behavior.
    """
    def __init__(self, patch_dim: int = 32 * 32, num_patches: int = 64, embed_dim: int = 768):
        super().__init__()
        self.patch_dim = patch_dim
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        # Linear projection for flattened patches -> embedding
        self.patch_projection = nn.Linear(self.patch_dim, self.embed_dim)

        # Learnable positional embeddings
        self.positional_embedding = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))

        # A simple classifier token projection (kept minimal)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # Small transformer encoder stub (one layer) to keep behavior predictable
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape (batch_size, num_patches, patch_dim)
        returns: tensor of shape (batch_size, num_patches + 1, embed_dim)
        """
        batch_size, patches, dim = x.shape
        assert patches == self.num_patches, f"Expected {self.num_patches} patches, got {patches}"
        assert dim == self.patch_dim, f"Expected patch dim {self.patch_dim}, got {dim}"

        # Project patches to embeddings
        x = self.patch_projection(x)  # (batch_size, num_patches, embed_dim)

        # Prepend cls token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches + 1, embed_dim)

        # Add positional embeddings (repeat positional embedding for cls token at front)
        pos_embed = torch.cat((torch.zeros(1, 1, self.embed_dim, device=self.positional_embedding.device),
                               self.positional_embedding), dim=1)
        x = x + pos_embed

        # Transformer expects shape (seq_len, batch, embed_dim)
        x = x.transpose(0, 1)
        x = self.encoder(x)
        x = x.transpose(0, 1)

        return x


# Example usage demonstrating the same reshape logic as the original snippet:

# Parameters (these mirror the original values)
image_size = 256
patch_size = 32
batch_size = 8

num_patches = compute_num_patches(image_size=image_size, patch_size=patch_size)  # 64
patch_dim = compute_patch_dim(patch_size=patch_size)  # 1024

# Create a flat input and reshape it to (batch_size, num_patches, patch_dim)
flat_input = torch.randn(batch_size, num_patches * patch_dim)
input_tensor = flat_input.view(batch_size, num_patches, patch_dim)

# Instantiate model and run forward
model = VisionTransformer(patch_dim=patch_dim, num_patches=num_patches)
output = model(input_tensor)

# Example shapes
print("Input shape:", input_tensor.shape)    # (batch_size, 64, 1024)
print("Output shape:", output.shape)         # (batch_size, 65, embed_dim)