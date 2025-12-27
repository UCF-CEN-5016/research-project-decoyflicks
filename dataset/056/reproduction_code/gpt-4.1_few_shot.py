import numpy as np
import keras
from keras import layers, ops

# Minimal Patches layer reproducing the error
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        # images dtype int32 causes error in extract_patches
        patches = ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                patches.shape[0],
                (images.shape[1] // self.patch_size) * (images.shape[2] // self.patch_size),
                -1,
            ),
        )
        return patches

# Sample data with dtype int32 (as in the reported error)
image_size = 72
patch_size = 6

# Create a dummy image tensor with dtype int32
image = np.random.randint(0, 256, size=(1, image_size, image_size, 3), dtype=np.int32)
image_tensor = ops.convert_to_tensor(image)

# Instantiate Patches layer and call it
patch_layer = Patches(patch_size)
patches = patch_layer(image_tensor)  # This line triggers the error

print(f"Patches shape: {patches.shape}")