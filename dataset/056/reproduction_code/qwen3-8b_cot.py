import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Assuming 'image' is a numpy array of type int32 (or other type)
resized_image = tf.image.resize(
    tf.convert_to_tensor(image.astype(np.float32)),  # Convert to float32
    size=(image_size, image_size)
)

class Patches(Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        images = tf.cast(images, tf.float32)  # Ensure input is float32
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patch_shape = (self.patch_size, self.patch_size, images.shape[3])
        patches = tf.reshape(patches, [images.shape[0], -1, self.patch_size * self.patch_size * images.shape[3]])
        return patches

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Example image (assume it's loaded as int32)
image = np.random.randint(0, 256, (256, 256, 3), dtype=np.int32)
image_size = 64  # Example size

# Resize and convert to float32
resized_image = tf.image.resize(
    tf.convert_to_tensor(image.astype(np.float32)),
    size=(image_size, image_size)
)

# Define Patches layer
class Patches(Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        images = tf.cast(images, tf.float32)
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patch_shape = (self.patch_size, self.patch_size, images.shape[3])
        patches = tf.reshape(patches, [images.shape[0], -1, self.patch_size * self.patch_size * images.shape[3]])
        return patches

# Use the layer
patches_layer = Patches(patch_size=16)
output = patches_layer(resized_image)