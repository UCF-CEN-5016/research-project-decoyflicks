self.to_patch_embedding = nn.Linear(patch_size * patch_size, dim)

print(input.shape)  # Should be (batch_size, num_patches, 1024)

import torch
import torch.nn as nn

class ViTModel(nn.Module):
    def __init__(self, patch_size=32, dim=1024):
        super(ViTModel, self).__init__()
        self.patch_size = patch_size
        self.to_patch_embedding = nn.Linear(patch_size * patch_size, dim)
        # Other layers...

    def forward(self, x):
        # Reshape input to (batch_size, num_patches, patch_size * patch_size)
        x = x.view(x.size(0), -1, self.patch_size * self.patch_size)
        x = self.to_patch_embedding(x)
        # Proceed with rest of the model...