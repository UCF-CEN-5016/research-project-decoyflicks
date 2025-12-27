import torch
import torch.nn as nn
import torch.nn.functional as F


class MPP(nn.Module):
    def __init__(self, dim: int = 1024, masking_rate: float = 0.5):
        super().__init__()
        self.dim = dim
        self.masking_rate = masking_rate

        self.transformer = ViT(dim=dim)
        num_patches = 8 * 8
        num_tokens = 1 + num_patches  # 1 for CLS token
        self.position_embeddings = nn.Parameter(torch.zeros(num_tokens, dim))

    def forward(self, x: torch.Tensor):
        """
        x: Tensor of shape [batch_size, num_tokens, dim]
        """
        batch_size = x.shape[0]
        num_patches = 8 * 8
        num_tokens = 1 + num_patches

        # Add positional embeddings
        pos = self.position_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos

        # Create a mask indicating which tokens are kept (1) vs masked (0).
        # masking_rate indicates fraction to mask, so keep prob = 1 - masking_rate.
        keep_prob = 1.0 - self.masking_rate
        device = x.device
        mask_vector = torch.bernoulli(torch.full((num_tokens,), keep_prob, device=device))
        # Ensure CLS token is always kept (unmasked)
        mask_vector[0] = 1.0
        mask = mask_vector.unsqueeze(0).expand(batch_size, -1).float()  # [B, num_tokens]

        # Apply masking during training; no masking during evaluation
        if self.training:
            x_masked = x * mask.unsqueeze(-1)
        else:
            x_masked = x

        # Extract image tokens (exclude CLS) for prediction/loss computation
        image_tokens = x_masked[:, 1 : 1 + num_patches, :]  # [B, num_patches, dim]

        # Project / reshape masked image tokens to expected shape before loss
        projected_tokens = image_tokens.view(batch_size, image_tokens.size(1), self.dim)  # [B, num_patches, dim]

        # Reconstruct with transformer and apply layer norm
        x_reconstructed = self.transformer(x_masked)  # [B, num_tokens, dim]
        x_reconstructed = F.layer_norm(x_reconstructed, (self.dim,))

        if self.training:
            loss = F.mse_loss(projected_tokens, image_tokens)
            return (loss, loss)
        else:
            return (x_reconstructed.mean(),)