import tensorflow as tf
from keras.layers import Patches, Image

# Set up the model
model = tf.keras.Sequential([
    Patches((28, 28)),
    Image()
])

# Load an image and resize it
image_size = (72, 72)
resized_image = tf.image.resize(tf.ones([1, 72, 72, 3]), image_size)

# Extract patches from the resized image
patches = model(resized_image)

print(patches.shape)  # Output: should be a tensor with shape (?, ?, ?, 3)