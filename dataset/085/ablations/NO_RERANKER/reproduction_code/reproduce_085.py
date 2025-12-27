import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a batch size and image dimensions
batch_size = 4
height, width = 224, 224

# Create a random tensor for input images
img = torch.randn(batch_size, 3, height, width)

# Define a dummy encoder class
class DummyEncoder:
    def __init__(self):
        num_patches = (height // 16) * (width // 16)
        self.pos_embedding = torch.randn(1, num_patches + 1, 768)

    def to_patch_embedding(self, x):
        return x.view(batch_size, -1, 768)  # Dummy implementation

encoder = DummyEncoder()

# Define the MAE class
class MAE(nn.Module):
    def __init__(self, encoder, decoder_dim, masking_ratio=0.75, decoder_depth=1, decoder_heads=8, decoder_dim_head=64):
        super().__init__()
        self.masking_ratio = masking_ratio
        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding, encoder.to_patch_embedding
        self.decoder_dim = decoder_dim
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)

    def forward(self, img):
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape
        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]  # Bug line
        return tokens

# Instantiate the MAE class
mae = MAE(encoder, decoder_dim=768)

# Call the forward method
try:
    output = mae(img)
    print("Output shape:", output.shape)
    assert not torch.isnan(output).any()
except Exception as e:
    print("Error:", e)