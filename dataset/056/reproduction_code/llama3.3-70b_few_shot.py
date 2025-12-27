import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Define patch size and image size
patch_size = 6
image_size = 72

# Create a sample image
image = np.random.randint(0, 256, size=(image_size, image_size, 3), dtype=np.int32)

# Resize the image
resized_image = tf.image.resize(tf.convert_to_tensor([image]), size=(image_size, image_size))

# Define the Patches layer
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        return patches

# Create an instance of the Patches layer
patches_layer = Patches(patch_size)

# Apply the Patches layer (this will cause the error)
patches = patches_layer(resized_image)

# To fix the error, cast the image to float32 before applying the Patches layer
resized_image_float = tf.cast(resized_image, tf.float32)
patches_fixed = patches_layer(resized_image_float)