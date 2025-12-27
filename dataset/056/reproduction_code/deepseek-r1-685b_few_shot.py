import keras
from keras import ops
import numpy as np
import matplotlib.pyplot as plt

# Minimal Patches layer implementation
class Patches(keras.layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        return keras.ops.image.extract_patches(images, size=self.patch_size)

# Simulate the error
image_size = 72
patch_size = 6

# Create test image (int32 dtype like original error)
test_image = np.random.randint(0, 256, (1, image_size, image_size, 3), dtype=np.int32)
test_image = ops.convert_to_tensor(test_image)

# This will fail with the same error
patches = Patches(patch_size)(test_image)

# Working version with float32 conversion
fixed_image = ops.cast(test_image, "float32")
patches = Patches(patch_size)(fixed_image)  # This works
print("Successfully extracted patches:", patches.shape)