import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Define the Patches class
class Patches(keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        num_patches_h = images.shape[1] // self.patch_size
        num_patches_w = images.shape[2] // self.patch_size
        patches = tf.image.extract_patches(images, sizes=[1, self.patch_size, self.patch_size, 1], strides=[1, self.patch_size, self.patch_size, 1], rates=[1, 1, 1, 1], padding='VALID')
        patches = tf.reshape(patches, (patches.shape[0], -1, patches.shape[-1]))
        return patches

# Set up a minimal environment
image_size = 72
patch_size = 6
x_train = tf.random.uniform((100, 224, 224, 3), minval=0, maxval=256, dtype=tf.int32)

# Add triggering conditions
image = x_train[np.random.choice(range(x_train.shape[0]))]
resized_image = tf.image.resize(tf.cast(image, tf.float32), (image_size, image_size))  # Cast to float32

# Create an instance of the Patches class
patches_layer = Patches(patch_size)

# Trigger the bug
patches = patches_layer(resized_image[tf.newaxis, ...])  # Add a batch dimension

# Print the patches shape
print(f"Patches shape: {patches.shape}")

# Plot the patches
n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy().astype("uint8"))
    plt.axis("off")
plt.show()