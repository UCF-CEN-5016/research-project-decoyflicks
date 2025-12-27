class MPP(nn.Module):
    def __init__(self, dim=1024, masking_rate=0.5):
        super(MPP, self).__init__()
        self.transformer = ViT(dim=dim)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1 + 8 * 8, dim)  # 1 for CLS token
        )
        
        self.masking_rate = masking_rate

    def forward(self, x):
        B = x.shape[0]
        masked_input = x
        
        # Shape: [batch_size, num_tokens (including CLS), dim]
        mask = torch.bernoulli(torch.ones(1 + 8 * 8, dtype=torch.float32),
                               self.masking_rate)
        pos = self.position_embeddings.unsqueeze(0).expand(B, -1, -1)
        x = pos + masked_input
        
        # CLS token is always unmasked
        mask = F.pad(mask.view(-1), (B, 0), mode='constant',
                     value=0)  # [batch_size * num_tokens]
        mask = mask.view(B, -1).float()  # [batch_size, num_tokens]
        
        # During training, randomly mask some tokens
        if self.training:
            x_masked = x * mask.unsqueeze(-1)
        else:
            x_masked = x
        
        # Extract masked tokens for prediction
        masked_tokens = x_masked.permute(0, 2, 1)  # [batch_size, dim, num_tokens]
        masked_tokens = masked_tokens[:, :, : (8 * 8)]
        
        # Project masked tokens to the correct dimension before layer norm
        projected_tokens = masked_tokens.view(-1, masked_tokens.size(2), 1024)
        
        x_reconstructed = self.transformer(x_masked)  # [batch_size, num_tokens, dim]
        x_reconstructed = F.layer_norm(x_reconstructed, [dim])
        
        if self.training:
            return (F.mse_loss(projected_tokens, masked_tokens),) * 2
        else:
            return (x_reconstructed.mean(), )