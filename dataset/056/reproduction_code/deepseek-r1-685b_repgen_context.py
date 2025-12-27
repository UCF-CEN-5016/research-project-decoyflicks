import tensorflow as tf
from tensorflow.keras.layers import Layer

# Minimal Patches layer implementation
class Patches(Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        return tf.image.extract_patches(images, sizes=[1, self.patch_size, self.patch_size, 1], strides=[1, self.patch_size, self.patch_size, 1], rates=[1, 1, 1, 1], padding='VALID')

# Simulate the error
image_size = 72
patch_size = 6

# Create test image (int32 dtype like original error)
test_image = tf.random.uniform((1, image_size, image_size, 3), minval=0, maxval=256, dtype=tf.int32)

# This will fail with the same error
patches = Patches(patch_size)(test_image)

# Working version with float32 conversion
fixed_image = tf.cast(test_image, tf.float32)
patches = Patches(patch_size)(fixed_image)  # This works
print("Successfully extracted patches:", patches.shape)