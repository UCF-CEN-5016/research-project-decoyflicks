import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Minimal Patches layer implementation
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        return tf.image.extract_patches(images, sizes=[1, self.patch_size, self.patch_size, 1], strides=[1, self.patch_size, self.patch_size, 1], rates=[1, 1, 1, 1], padding='VALID')

# Simulate the error
image_size = 72
patch_size = 6

# Create test image (int32 dtype like original error)
test_image = np.random.randint(0, 256, (1, image_size, image_size, 3), dtype=np.int32)
test_image = tf.convert_to_tensor(test_image, dtype=tf.int32)

# This will fail with the same error
patches = Patches(patch_size)(test_image)

# Working version with float32 conversion
fixed_image = tf.cast(test_image, tf.float32)
patches = Patches(patch_size)(fixed_image)  # This works
print("Successfully extracted patches:", patches.shape)