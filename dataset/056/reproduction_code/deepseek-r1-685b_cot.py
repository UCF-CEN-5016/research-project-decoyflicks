import numpy as np
import keras
from keras import ops
import matplotlib.pyplot as plt

# Minimal Patches layer reproduction
class Patches(keras.layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = ops.shape(images)[0]
        patches = keras.ops.image.extract_patches(
            images, size=self.patch_size
        )
        return patches

# Create test image (uint8 -> int32 after tensor conversion)
image = np.random.randint(0, 256, (72, 72, 3), dtype=np.uint8
resized_image = ops.convert_to_tensor([image])  # becomes int32 tensor

# This will fail with the same error
# patches = Patches(patch_size=6)(resized_image)

# Solution: Convert to float32 before patch extraction
resized_image_float = ops.cast(resized_image, "float32")
patches = Patches(patch_size=6)(resized_image_float)  # Works!

print("Patches extracted successfully!")
print("Patch shape:", patches.shape)

patches = Patches(patch_size)(ops.cast(resized_image, "float32"))