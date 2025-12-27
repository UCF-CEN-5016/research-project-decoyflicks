class VisionTransformer(nn.Module):
    def __init__(self, dim=32*32, num_patches=64, ...):
        super().__init__()
        self.patch_dim = dim
        # ...

def calculate_num_patches(image_size, patch_size):
    return (image_size // patch_size) ** 2

def calculate_patch_dim(patch_size):
    return patch_size ** 2

input_shape = (batch_size, num_patches, patch_dim)
num_patches = calculate_num_patches(image_size=256, patch_size=32)
patch_dim = calculate_patch_dim(patch_size=32)

# Example: Reshape [batch, 3072] to [batch, 64, 1024]
input_tensor = input_tensor.view(batch_size, num_patches, patch_dim)