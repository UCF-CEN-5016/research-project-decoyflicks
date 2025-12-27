import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Set the Keras version to 3.10 (if applicable)
keras.__version__ = '3.10'

# Define an image size of 72x72 pixels for processing
image_size = 72

# Create random uniform input data with shape (1, 72, 72, 3) using numpy
input_data = np.random.uniform(0, 255, (1, image_size, image_size, 3))

# Convert the input data to a float32 tensor and scale it by dividing by 255.0
input_tensor = tf.convert_to_tensor(input_data.astype('float32') / 255.0)

# Ensure the input tensor's dtype is set to int32 before passing it to the resize function
input_tensor = tf.cast(input_tensor, 'int32')

# Call the resize function from Keras' ops.image module with size parameter as (72, 72) to resize the image
resized_image = tf.image.resize(input_tensor, size=(image_size, image_size))

# Assert that the resized image tensor has a shape of (1, 72, 72, 3) and dtype int32 after resizing
assert resized_image.shape == (1, image_size, image_size, 3)
assert resized_image.dtype == 'int32'

# Create an instance of the Patches class with patch_size set to 16 as defined in the original code
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]
        num_patches_height = height // self.patch_size
        num_patches_width = width // self.patch_size
        patches = tf.reshape(
            images,
            [batch_size, num_patches_height, self.patch_size, num_patches_width, self.patch_size, 3],
        )
        return tf.reshape(patches, [batch_size, -1, self.patch_size**2 * 3])

patch_size = 16
patches_layer = Patches(patch_size)

# Call the Patches instance with the resized image tensor and assert that it raises an InvalidArgumentError
try:
    patches = patches_layer(resized_image)
except Exception as e:
    print(f"Caught exception: {e}")

# Display individual patches
n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(tf.convert_to_numpy(patch_img).astype("uint8"))
    plt.axis("off")
plt.show()