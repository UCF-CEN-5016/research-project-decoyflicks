import torch
from modeling import MAE

# Initialize model with appropriate number of positions and token dimensions
mae = MAE()
pos_embedding = mae.encoder.pos_embedding  # Shape: (max_pos, d)
tokens = torch.randn(10)                    # Shape: (10,) 

num_patches = 5
if len(pos_embedding) > num_patches + 1:
    pos_slice = pos_embedding[:, 1:(num_patches+1)]
else:
    raise IndexError("Not enough positional embeddings available.")

print("Shapes after slicing:", tokens.shape, pos_slice.shape)

# Ensure shapes are compatible for addition (broadcasting)
tokens = tokens.unsqueeze(-1)                # Shape: (10, 1)
pos_slice = pos_slice.unsqueeze(0)          # Shape: (1, d)
combined_features = tokens + pos_slice     # Broadcasting occurs here

print("Combined features shape:", combined_features.shape)