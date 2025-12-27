import torch
from dalle_pytorch import DALLE

# Minimal DALL-e model setup
dalle = DALLE(
    dim=512,
    vae=None,  # Normally would be a VAE model
    num_text_tokens=10000,
    text_seq_len=256,
    depth=2,
    heads=8
)

# Sample inputs matching notebook's expected format
text = torch.randint(0, 10000, (4, 256))  # 4 captions of length 256
image_codes = torch.randn(4, 32, 32)      # 4 fake image encodings
mask = torch.ones(4, 256).bool()          # Attention mask

# This will reproduce the error
try:
    loss = dalle(text, image_codes, mask=mask, return_loss=True)
except TypeError as e:
    print(f"Error occurred: {e}")
    print("The model's forward() method doesn't accept 'mask' argument")