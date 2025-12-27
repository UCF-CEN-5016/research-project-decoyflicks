import torch
import torch.nn as nn
import torch.nn.functional as F

class DummyEncoder:
    def __init__(self):
        self.pos_embedding = nn.Parameter(torch.randn(1, 197, 768))  # 196 patches + 1 for cls token
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(3 * 16 * 16, 768)  # Assuming patch size of 16x16
        )
    
    def transformer(self, tokens):
        return tokens  # Dummy transformer that returns tokens as is

class MAE(nn.Module):
    def __init__(self, encoder, decoder_dim, masking_ratio=0.75, decoder_depth=1, decoder_heads=8, decoder_dim_head=64):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio
        self.encoder = encoder
        num_patches = (224 // 16) * (224 // 16)
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, 3 * 16 * 16)

    def forward(self, img):
        device = img.device
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape
        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        batch_range = torch.arange(batch, device=device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]
        masked_patches = patches[batch_range, masked_indices]
        encoded_tokens = self.encoder.transformer(tokens)
        decoder_tokens = self.to_pixels(encoded_tokens)
        recon_loss = F.mse_loss(decoder_tokens, masked_patches)
        return recon_loss

batch_size = 4
img = torch.randn(batch_size, 3, 224, 224)
encoder = DummyEncoder()
mae = MAE(encoder, decoder_dim=768)
loss = mae(img)
print(loss)