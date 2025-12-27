input_shape = (batch_size, num_patches, patch_dim)

# If image is 256x256 and patch size is 32
num_patches = (256 // 32) * (256 // 32) = 64
patch_dim = 32 * 32 = 1024
input_shape = (batch_size, 64, 1024)

class VisionTransformer(nn.Module):
    def __init__(self, dim=1024, num_patches=64, ...):
        super().__init__()
        self.patch_dim = dim
        # ...

# Example: Reshape [batch, 3072] to [batch, 64, 1024]
input_tensor = input_tensor.view(batch_size, 64, 1024)

# Example of patch embedding
patch_size = 32
image_size = 256
num_patches = (image_size // patch_size) ** 2
patch_dim = patch_size ** 2