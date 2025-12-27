import torch
import torch.nn as nn
import torch.nn.functional as F

class MockEncoder:
    def __init__(self, num_patches, encoder_dim):
        self.pos_embedding = torch.randn(1, num_patches + 1, encoder_dim)
        self.to_patch_embedding = nn.Identity()
        self.transformer = nn.Identity()

class MAE(nn.Module):
    def __init__(self, encoder, decoder_dim, masking_ratio=0.75):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio
        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = nn.Identity()
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, img):
        device = img.device
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape
        tokens = self.patch_to_emb(patches)
        # Fixed the syntax error by adding the missing closing bracket
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        batch_range = torch.arange(batch, device=device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]
        masked_patches = patches[batch_range, masked_indices]
        encoded_tokens = self.encoder.transformer(tokens)
        decoder_tokens = self.enc_to_dec(encoded_tokens)
        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)
        mask_tokens = self.mask_token.unsqueeze(0).expand(batch, -1)  # Replacing repeat with expand
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)
        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens)
        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)
        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        return recon_loss

image_size = 224
patch_size = 16
num_patches = (image_size // patch_size) * (image_size // patch_size)
batch_size = 4
encoder_dim = 768

mock_encoder = MockEncoder(num_patches, encoder_dim)
mae_model = MAE(mock_encoder, decoder_dim=encoder_dim, masking_ratio=0.75)
input_tensor = torch.randn(batch_size, 3, image_size, image_size)

output_loss = mae_model(input_tensor)
assert isinstance(output_loss, torch.Tensor)
assert output_loss.shape == ()
print(output_loss.item())