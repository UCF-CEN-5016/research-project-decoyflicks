import torch
from dalle_pytorch import DALLE

# Set up a minimal DALL-e model
dalle = DALLE(
    dim=512,
    vae=None,  # Normally would be a VAE model
    num_text_tokens=10000,
    text_seq_len=256,
    depth=2,
    heads=8
)

# Prepare sample inputs in the expected format
text = torch.randint(0, 10000, (4, 256))  # 4 captions of length 256
image_codes = torch.randn(4, 32, 32)      # 4 fake image encodings
mask = torch.ones(4, 256, dtype=torch.bool)  # Attention mask

# Verify if the model's forward() method accepts 'mask' argument
forward_signature = dalle.forward.__code__.co_varnames
if 'mask' in forward_signature:
    try:
        loss = dalle(text, image_codes, mask=mask, return_loss=True)
    except TypeError as e:
        print(f"Error occurred: {e}")
else:
    print("The model's forward() method doesn't accept 'mask' argument")